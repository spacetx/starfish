import codecs
import io
import pickle
import tarfile
import warnings
from dataclasses import dataclass
from typing import BinaryIO, List, Mapping, MutableSequence, Optional, Tuple, Type

import numpy as np
from packaging.version import Version

from starfish.core.errors import DataFormatWarning
from starfish.core.types import ArrayLike, Axes, Coordinates, Number, STARFISH_EXTRAS_KEY
from starfish.core.util.logging import Log
from . import binary_mask as bm


class AttrKeys:
    DOCTYPE = f"{STARFISH_EXTRAS_KEY}.DOCTYPE"
    VERSION = f"{STARFISH_EXTRAS_KEY}.VERSION"


DOCTYPE_STRING = "starfish/BinaryMaskCollection"


class BinaryMaskIO:
    ASCENDING_VERSIONS: List[Tuple[Version, Type["BinaryMaskIO"]]] = []
    VERSION: Version

    def __init_subclass__(cls, version_descriptor: Version, **kwargs):
        super.__init_subclass__(**kwargs)  # type: ignore
        cls.VERSION = version_descriptor
        BinaryMaskIO.ASCENDING_VERSIONS.append((version_descriptor, cls))
        BinaryMaskIO.ASCENDING_VERSIONS.sort()

    @staticmethod
    def read_versioned_binary_mask(
            file_obj: BinaryIO,
    ) -> bm.BinaryMaskCollection:
        with tarfile.open(mode="r|gz", fileobj=file_obj) as tf:
            if tf.pax_headers is None:
                raise ValueError(
                    "file does not appear to be a binary mask file (does not have headers)")
            if tf.pax_headers.get(AttrKeys.DOCTYPE, None) != DOCTYPE_STRING:
                raise ValueError(
                    "file does not appear to be a binary mask file (missing or incorrect doctype)")
            if AttrKeys.VERSION not in tf.pax_headers:
                raise ValueError("file missing version data")

            requested_version = Version(tf.pax_headers[AttrKeys.VERSION])

            for version, implementation in BinaryMaskIO.ASCENDING_VERSIONS:
                if version == requested_version:
                    return implementation().read_binary_mask(tf)
            else:
                raise ValueError(f"No reader for version {requested_version}")

    @staticmethod
    def write_versioned_binary_mask(
            file_obj: BinaryIO,
            binary_mask: bm.BinaryMaskCollection,
            requested_version: Optional[Version] = None,
    ):
        if requested_version is None:
            implementation = BinaryMaskIO.ASCENDING_VERSIONS[-1][1]
        else:
            for version, implementing_class in BinaryMaskIO.ASCENDING_VERSIONS:
                if version == requested_version:
                    implementation = implementing_class
                    break
            else:
                raise ValueError(f"No writer for version {requested_version}")

        implementation().write_binary_mask(file_obj, binary_mask)

    def read_binary_mask(self, tf: tarfile.TarFile) -> bm.BinaryMaskCollection:
        raise NotImplementedError()

    def write_binary_mask(self, file_obj: BinaryIO, binary_mask: bm.BinaryMaskCollection):
        raise NotImplementedError()


class v0_0(BinaryMaskIO, version_descriptor=Version("0.0")):
    LOG_FILENAME = "log"
    PIXEL_TICKS_FILE = "pixel_ticks.pickle"
    PHYSICAL_TICKS_FILE = "physical_ticks.pickle"
    MASK_PREFIX = "masks/"

    @dataclass
    class MaskOnDisk:
        binary_mask: np.ndarray
        offsets: Tuple[int, ...]

    def read_binary_mask(self, tf: tarfile.TarFile) -> bm.BinaryMaskCollection:
        log: Optional[Log] = None
        masks: MutableSequence[bm.MaskData] = []
        pixel_ticks: Optional[Mapping[Axes, ArrayLike[int]]] = None
        physical_ticks: Optional[Mapping[Coordinates, ArrayLike[Number]]] = None

        while True:
            tarinfo: Optional[tarfile.TarInfo] = tf.next()
            if tarinfo is None:
                break

            # wrap it in a BytesIO object to ensure we never seek backwards.
            extracted_fh = tf.extractfile(tarinfo)
            if extracted_fh is None:
                raise ValueError(f"Unable to extract file {tarinfo.name}")
            byte_stream = io.BytesIO(extracted_fh.read())

            if tarinfo.name == v0_0.LOG_FILENAME:
                string_stream = codecs.getreader("utf-8")(byte_stream)
                log = Log.decode(string_stream.read())
            elif tarinfo.name == v0_0.PIXEL_TICKS_FILE:
                pixel_ticks = pickle.load(byte_stream)
            elif tarinfo.name == v0_0.PHYSICAL_TICKS_FILE:
                physical_ticks = pickle.load(byte_stream)
            elif tarinfo.name.startswith(v0_0.MASK_PREFIX):
                mask_on_disk: v0_0.MaskOnDisk = pickle.load(byte_stream)
                if not isinstance(mask_on_disk, v0_0.MaskOnDisk):
                    raise TypeError("mask does not conform to expected mask structure")
                masks.append(bm.MaskData(mask_on_disk.binary_mask, mask_on_disk.offsets, None))
            else:
                warnings.warn(
                    f"Unexpected file in binary mask collection {tarinfo.name}",
                    DataFormatWarning
                )

        if pixel_ticks is None:
            raise ValueError("pixel coordinates not found")
        if physical_ticks is None:
            raise ValueError("physical coordinates not found")

        return bm.BinaryMaskCollection(pixel_ticks, physical_ticks, masks, log)

    def write_binary_mask(self, file_obj: BinaryIO, binary_mask: bm.BinaryMaskCollection):
        pax_headers: Mapping[str, str] = {
            AttrKeys.DOCTYPE: DOCTYPE_STRING,
            AttrKeys.VERSION: str(v0_0.VERSION),
        }
        with tarfile.open(
                fileobj=file_obj,
                mode="w|gz",
                format=tarfile.PAX_FORMAT,
                pax_headers=pax_headers,
        ) as tf:
            # write the log
            log_bytes = binary_mask._log.encode().encode("utf-8")
            write_to_tarfile(tf, v0_0.LOG_FILENAME, log_bytes)

            # write the pixel ticks
            pixel_ticks_bytes = pickle.dumps(binary_mask._pixel_ticks)
            write_to_tarfile(tf, v0_0.PIXEL_TICKS_FILE, pixel_ticks_bytes)

            # write the physical coordinate ticks
            physical_ticks_bytes = pickle.dumps(binary_mask._physical_ticks)
            write_to_tarfile(tf, v0_0.PHYSICAL_TICKS_FILE, physical_ticks_bytes)

            for ix, mask in binary_mask._masks.items():
                mask_on_disk = v0_0.MaskOnDisk(mask.binary_mask, mask.offsets)
                mask_bytes = pickle.dumps(mask_on_disk)
                write_to_tarfile(tf, f"{v0_0.MASK_PREFIX}{ix}", mask_bytes)


def write_to_tarfile(tf: tarfile.TarFile, name: str, data: bytes):
    tarinfo = tarfile.TarInfo(name=name)
    tarinfo.size = len(data)
    tf.addfile(tarinfo, io.BytesIO(data))
