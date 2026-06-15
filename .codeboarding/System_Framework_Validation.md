```mermaid
graph LR
    Validation_Schema_Enforcement["Validation & Schema Enforcement"]
    Core_Infrastructure_Configuration["Core Infrastructure & Configuration"]
    Algorithm_Framework_Type_System["Algorithm Framework & Type System"]
    Diagnostic_Simulation_Engine["Diagnostic & Simulation Engine"]
    Core_Infrastructure_Configuration -- "Provides configuration settings to" --> Validation_Schema_Enforcement
    Algorithm_Framework_Type_System -- "Injects metadata into" --> Core_Infrastructure_Configuration
    Validation_Schema_Enforcement -- "Validates output of" --> Diagnostic_Simulation_Engine
    Diagnostic_Simulation_Engine -- "Uses contracts to run simulations against" --> Algorithm_Framework_Type_System
    Validation_Schema_Enforcement -- "calls" --> Core_Infrastructure_Configuration
```

[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/CodeBoarding)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/diagrams)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)

## Details

Provides infrastructure including CLI routing, schema validation, and global configuration management.

### Validation & Schema Enforcement
Ensures input datasets comply with the SpaceTx JSON schema and acts as a gatekeeper for data integrity.


**Related Classes/Methods**:

- `starfish.core.spacetx_format.util.SpaceTxValidator`:27-203
- `starfish.core.spacetx_format.util.ExperimentValidator`:455-462
- `starfish.core.spacetx_format.cli.validate`:30-31



**Source Files:**

- [`starfish/core/spacetx_format/cli.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/cli.py)
  - `starfish.core.spacetx_format.cli.DefaultGroup` ([L9-L23](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/cli.py#L9-L23)) - Class
  - `starfish.core.spacetx_format.cli.DefaultGroup.parse_args` ([L14-L23](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/cli.py#L14-L23)) - Method
  - `starfish.core.spacetx_format.cli.validate` ([L30-L31](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/cli.py#L30-L31)) - Function
  - `starfish.core.spacetx_format.cli.codebook` ([L38-L48](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/cli.py#L38-L48)) - Function
  - `starfish.core.spacetx_format.cli.experiment` ([L55-L65](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/cli.py#L55-L65)) - Function
  - `starfish.core.spacetx_format.cli.fov` ([L72-L82](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/cli.py#L72-L82)) - Function
  - `starfish.core.spacetx_format.cli.manifest` ([L89-L99](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/cli.py#L89-L99)) - Function
  - `starfish.core.spacetx_format.cli.xarray` ([L105-L137](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/cli.py#L105-L137)) - Function
- [`starfish/core/spacetx_format/util.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py)
  - `starfish.core.spacetx_format.util._get_absolute_schema_path` ([L22-L24](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L22-L24)) - Function
  - `starfish.core.spacetx_format.util.SpaceTxValidator` ([L27-L203](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L27-L203)) - Class
  - `starfish.core.spacetx_format.util.SpaceTxValidator.__init__` ([L29-L39](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L29-L39)) - Method
  - `starfish.core.spacetx_format.util.SpaceTxValidator._create_validator` ([L42-L65](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L42-L65)) - Method
  - `starfish.core.spacetx_format.util.SpaceTxValidator.load_json` ([L68-L70](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L68-L70)) - Method
  - `starfish.core.spacetx_format.util.SpaceTxValidator._recurse_through_errors` ([L73-L104](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L73-L104)) - Method
  - `starfish.core.spacetx_format.util.SpaceTxValidator.validate_file` ([L106-L134](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L106-L134)) - Method
  - `starfish.core.spacetx_format.util.SpaceTxValidator.validate_object` ([L136-L174](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L136-L174)) - Method
  - `starfish.core.spacetx_format.util.SpaceTxValidator.fuzz_object` ([L176-L203](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L176-L203)) - Method
  - `starfish.core.spacetx_format.util.Fuzzer` ([L206-L308](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L206-L308)) - Class
  - `starfish.core.spacetx_format.util.Fuzzer.__init__` ([L208-L224](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L208-L224)) - Method
  - `starfish.core.spacetx_format.util.Fuzzer.fuzz` ([L226-L241](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L226-L241)) - Method
  - `starfish.core.spacetx_format.util.Fuzzer.state` ([L243-L269](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L243-L269)) - Method
  - `starfish.core.spacetx_format.util.Fuzzer.descend` ([L271-L308](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L271-L308)) - Method
  - `starfish.core.spacetx_format.util.Checker` ([L310-L346](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L310-L346)) - Class
  - `starfish.core.spacetx_format.util.Checker.LETTER` ([L313-L314](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L313-L314)) - Method
  - `starfish.core.spacetx_format.util.Checker.check` ([L316-L343](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L316-L343)) - Method
  - `starfish.core.spacetx_format.util.Checker.handle` ([L345-L346](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L345-L346)) - Method
  - `starfish.core.spacetx_format.util.Add` ([L348-L360](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L348-L360)) - Class
  - `starfish.core.spacetx_format.util.Add.LETTER` ([L351-L352](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L351-L352)) - Method
  - `starfish.core.spacetx_format.util.Add.handle` ([L354-L360](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L354-L360)) - Method
  - `starfish.core.spacetx_format.util.Del` ([L362-L372](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L362-L372)) - Class
  - `starfish.core.spacetx_format.util.Del.LETTER` ([L365-L366](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L365-L366)) - Method
  - `starfish.core.spacetx_format.util.Del.handle` ([L368-L372](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L368-L372)) - Method
  - `starfish.core.spacetx_format.util.Change` ([L374-L388](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L374-L388)) - Class
  - `starfish.core.spacetx_format.util.Change.LETTER` ([L377-L378](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L377-L378)) - Method
  - `starfish.core.spacetx_format.util.Change.__init__` ([L380-L382](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L380-L382)) - Method
  - `starfish.core.spacetx_format.util.Change.handle` ([L384-L388](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L384-L388)) - Method
  - `starfish.core.spacetx_format.util.get_schema_path` ([L391-L442](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L391-L442)) - Function
  - `starfish.core.spacetx_format.util.CodebookValidator` ([L445-L452](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L445-L452)) - Class
  - `starfish.core.spacetx_format.util.CodebookValidator.__init__` ([L451-L452](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L451-L452)) - Method
  - `starfish.core.spacetx_format.util.ExperimentValidator` ([L455-L462](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L455-L462)) - Class
  - `starfish.core.spacetx_format.util.ExperimentValidator.__init__` ([L461-L462](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L461-L462)) - Method
  - `starfish.core.spacetx_format.util.FOVValidator` ([L465-L472](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L465-L472)) - Class
  - `starfish.core.spacetx_format.util.FOVValidator.__init__` ([L471-L472](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L471-L472)) - Method
  - `starfish.core.spacetx_format.util.ManifestValidator` ([L475-L482](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L475-L482)) - Class
  - `starfish.core.spacetx_format.util.ManifestValidator.__init__` ([L481-L482](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/util.py#L481-L482)) - Method
- [`starfish/core/spacetx_format/validate_sptx.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/validate_sptx.py)
  - `starfish.core.spacetx_format.validate_sptx.validate_sptx` ([L15-L16](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/validate_sptx.py#L15-L16)) - Function
  - `starfish.core.spacetx_format.validate_sptx.validate` ([L18-L76](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/validate_sptx.py#L18-L76)) - Function
  - `starfish.core.spacetx_format.validate_sptx.validate_file` ([L79-L128](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/spacetx_format/validate_sptx.py#L79-L128)) - Function


### Core Infrastructure & Configuration
Manages global state, environment settings, CLI routing, and configuration context for the library.


**Related Classes/Methods**:

- `starfish.core.util.config.Config`:22-102
- `starfish.core._version.get_versions`:640-683



**Source Files:**

- [`starfish/core/_version.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py)
  - `starfish.core._version.get_keywords` ([L23-L33](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L23-L33)) - Function
  - `starfish.core._version.VersioneerConfig` ([L36-L44](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L36-L44)) - Class
  - `starfish.core._version.get_config` ([L47-L58](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L47-L58)) - Function
  - `starfish.core._version.NotThisMethod` ([L61-L62](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L61-L62)) - Class
  - `starfish.core._version.register_vcs_handler` ([L69-L77](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L69-L77)) - Function
  - `starfish.core._version.register_vcs_handler.decorate` ([L71-L76](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L71-L76)) - Function
  - `starfish.core._version.run_command` ([L80-L125](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L80-L125)) - Function
  - `starfish.core._version.versions_from_parentdir` ([L128-L153](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L128-L153)) - Function
  - `starfish.core._version.git_get_keywords` ([L157-L181](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L157-L181)) - Function
  - `starfish.core._version.git_versions_from_keywords` ([L185-L249](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L185-L249)) - Function
  - `starfish.core._version.git_pieces_from_vcs` ([L253-L387](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L253-L387)) - Function
  - `starfish.core._version.plus_or_dot` ([L390-L394](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L390-L394)) - Function
  - `starfish.core._version.render_pep440` ([L397-L419](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L397-L419)) - Function
  - `starfish.core._version.render_pep440_branch` ([L422-L449](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L422-L449)) - Function
  - `starfish.core._version.pep440_split_post` ([L452-L459](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L452-L459)) - Function
  - `starfish.core._version.render_pep440_pre` ([L462-L483](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L462-L483)) - Function
  - `starfish.core._version.render_pep440_post` ([L486-L510](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L486-L510)) - Function
  - `starfish.core._version.render_pep440_post_branch` ([L513-L539](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L513-L539)) - Function
  - `starfish.core._version.render_pep440_old` ([L542-L561](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L542-L561)) - Function
  - `starfish.core._version.render_git_describe` ([L564-L581](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L564-L581)) - Function
  - `starfish.core._version.render_git_describe_long` ([L584-L601](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L584-L601)) - Function
  - `starfish.core._version.render` ([L604-L637](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L604-L637)) - Function
  - `starfish.core._version.get_versions` ([L640-L683](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/_version.py#L640-L683)) - Function
- [`starfish/core/config/__init__.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py)
  - `starfish.core.config.__init__.special_prefix` ([L7-L15](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L7-L15)) - Function
  - `starfish.core.config.__init__.environ` ([L18-L53](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L18-L53)) - Class
  - `starfish.core.config.__init__.environ.__init__` ([L32-L33](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L32-L33)) - Method
  - `starfish.core.config.__init__.environ.__enter__` ([L35-L46](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L35-L46)) - Method
  - `starfish.core.config.__init__.environ.__exit__` ([L48-L53](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L48-L53)) - Method
  - `starfish.core.config.__init__.StarfishConfig` ([L56-L198](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L56-L198)) - Class
  - `starfish.core.config.__init__.StarfishConfig.__init__` ([L107-L149](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L107-L149)) - Method
  - `starfish.core.config.__init__.StarfishConfig._slicedimage_update` ([L151-L174](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L151-L174)) - Method
  - `starfish.core.config.__init__.StarfishConfig.flag` ([L176-L186](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L176-L186)) - Method
  - `starfish.core.config.__init__.StarfishConfig.slicedimage` ([L189-L190](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L189-L190)) - Method
  - `starfish.core.config.__init__.StarfishConfig.strict` ([L193-L194](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L193-L194)) - Method
  - `starfish.core.config.__init__.StarfishConfig.verbose` ([L197-L198](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/config/__init__.py#L197-L198)) - Method
- [`starfish/core/image/Filter/zero_by_channel_magnitude.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/zero_by_channel_magnitude.py)
  - `starfish.core.image.Filter.zero_by_channel_magnitude.ZeroByChannelMagnitude` ([L13-L94](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/zero_by_channel_magnitude.py#L13-L94)) - Class
  - `starfish.core.image.Filter.zero_by_channel_magnitude.ZeroByChannelMagnitude.__init__` ([L28-L31](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/zero_by_channel_magnitude.py#L28-L31)) - Method
  - `starfish.core.image.Filter.zero_by_channel_magnitude.ZeroByChannelMagnitude.run` ([L35-L94](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/Filter/zero_by_channel_magnitude.py#L35-L94)) - Method
- [`starfish/core/image/_registration/ApplyTransform/warp.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/ApplyTransform/warp.py)
  - `starfish.core.image._registration.ApplyTransform.warp.Warp` ([L17-L63](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/ApplyTransform/warp.py#L17-L63)) - Class
  - `starfish.core.image._registration.ApplyTransform.warp.Warp.run` ([L42-L63](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/ApplyTransform/warp.py#L42-L63)) - Method
  - `starfish.core.image._registration.ApplyTransform.warp.warp` ([L66-L87](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/ApplyTransform/warp.py#L66-L87)) - Function
- [`starfish/core/image/_registration/LearnTransform/translation.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/LearnTransform/translation.py)
  - `starfish.core.image._registration.LearnTransform.translation.Translation.__init__` ([L36-L42](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/LearnTransform/translation.py#L36-L42)) - Method
  - `starfish.core.image._registration.LearnTransform.translation.Translation.run` ([L44-L90](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/LearnTransform/translation.py#L44-L90)) - Method
- [`starfish/core/image/_registration/transforms_list.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/transforms_list.py)
  - `starfish.core.image._registration.transforms_list.TransformsList` ([L24-L166](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/transforms_list.py#L24-L166)) - Class
  - `starfish.core.image._registration.transforms_list.TransformsList.__init__` ([L28-L45](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/transforms_list.py#L28-L45)) - Method
  - `starfish.core.image._registration.transforms_list.TransformsList.__repr__` ([L47-L53](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/transforms_list.py#L47-L53)) - Method
  - `starfish.core.image._registration.transforms_list.TransformsList.append` ([L55-L73](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/transforms_list.py#L55-L73)) - Method
  - `starfish.core.image._registration.transforms_list.TransformsList._verify_version` ([L76-L82](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/transforms_list.py#L76-L82)) - Method
  - `starfish.core.image._registration.transforms_list.TransformsList.to_dict` ([L84-L110](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/transforms_list.py#L84-L110)) - Method
  - `starfish.core.image._registration.transforms_list.TransformsList.to_json` ([L112-L122](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/transforms_list.py#L112-L122)) - Method
  - `starfish.core.image._registration.transforms_list.TransformsList.from_dict` ([L125-L145](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/transforms_list.py#L125-L145)) - Method
  - `starfish.core.image._registration.transforms_list.TransformsList.from_json` ([L148-L166](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/image/_registration/transforms_list.py#L148-L166)) - Method
- [`starfish/core/util/click/__init__.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/click/__init__.py)
  - `starfish.core.util.click.__init__.RequiredParentOption` ([L17-L54](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/click/__init__.py#L17-L54)) - Class
  - `starfish.core.util.click.__init__.RequiredParentOption.handle_parse_result` ([L23-L54](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/click/__init__.py#L23-L54)) - Method
  - `starfish.core.util.click.__init__.option` ([L57-L59](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/click/__init__.py#L57-L59)) - Function
- [`starfish/core/util/clock.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/clock.py)
  - `starfish.core.util.clock.timeit` ([L6-L10](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/clock.py#L6-L10)) - Function
- [`starfish/core/util/config.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/config.py)
  - `starfish.core.util.config.NestedDict` ([L6-L19](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/config.py#L6-L19)) - Class
  - `starfish.core.util.config.NestedDict.__missing__` ([L8-L10](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/config.py#L8-L10)) - Method
  - `starfish.core.util.config.NestedDict.update` ([L12-L19](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/config.py#L12-L19)) - Method
  - `starfish.core.util.config.Config` ([L22-L102](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/config.py#L22-L102)) - Class
  - `starfish.core.util.config.Config.__init__` ([L26-L57](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/config.py#L26-L57)) - Method
  - `starfish.core.util.config.Config.lookup` ([L59-L102](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/config.py#L59-L102)) - Method
- [`starfish/core/util/exec.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/exec.py)
  - `starfish.core.util.exec.stages` ([L9-L70](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/exec.py#L9-L70)) - Function
  - `starfish.core.util.exec.stages.callback` ([L48-L49](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/exec.py#L48-L49)) - Function
  - `starfish.core.util.exec.prepare_stage` ([L73-L112](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/util/exec.py#L73-L112)) - Function
- [`versioneer.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py)
  - `versioneer.VersioneerConfig` ([L333-L342](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L333-L342)) - Class
  - `versioneer.get_root` ([L345-L391](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L345-L391)) - Function
  - `versioneer.get_config_from_root` ([L394-L439](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L394-L439)) - Function
  - `versioneer.NotThisMethod` ([L442-L443](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L442-L443)) - Class
  - `versioneer.register_vcs_handler` ([L451-L457](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L451-L457)) - Function
  - `versioneer.register_vcs_handler.decorate` ([L453-L456](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L453-L456)) - Function
  - `versioneer.run_command` ([L460-L505](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L460-L505)) - Function
  - `versioneer.git_get_keywords` ([L1195-L1219](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1195-L1219)) - Function
  - `versioneer.git_versions_from_keywords` ([L1223-L1287](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1223-L1287)) - Function
  - `versioneer.git_pieces_from_vcs` ([L1291-L1425](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1291-L1425)) - Function
  - `versioneer.do_vcs_install` ([L1428-L1463](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1428-L1463)) - Function
  - `versioneer.versions_from_parentdir` ([L1466-L1491](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1466-L1491)) - Function
  - `versioneer.versions_from_file` ([L1512-L1526](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1512-L1526)) - Function
  - `versioneer.write_to_version_file` ([L1529-L1536](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1529-L1536)) - Function
  - `versioneer.plus_or_dot` ([L1539-L1543](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1539-L1543)) - Function
  - `versioneer.render_pep440` ([L1546-L1568](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1546-L1568)) - Function
  - `versioneer.render_pep440_branch` ([L1571-L1598](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1571-L1598)) - Function
  - `versioneer.pep440_split_post` ([L1601-L1608](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1601-L1608)) - Function
  - `versioneer.render_pep440_pre` ([L1611-L1632](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1611-L1632)) - Function
  - `versioneer.render_pep440_post` ([L1635-L1659](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1635-L1659)) - Function
  - `versioneer.render_pep440_post_branch` ([L1662-L1688](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1662-L1688)) - Function
  - `versioneer.render_pep440_old` ([L1691-L1710](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1691-L1710)) - Function
  - `versioneer.render_git_describe` ([L1713-L1730](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1713-L1730)) - Function
  - `versioneer.render_git_describe_long` ([L1733-L1750](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1733-L1750)) - Function
  - `versioneer.render` ([L1753-L1786](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1753-L1786)) - Function
  - `versioneer.VersioneerBadRootError` ([L1789-L1790](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1789-L1790)) - Class
  - `versioneer.get_versions` ([L1793-L1866](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1793-L1866)) - Function
  - `versioneer.get_version` ([L1869-L1871](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1869-L1871)) - Function
  - `versioneer.get_cmdclass` ([L1874-L2121](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1874-L2121)) - Function
  - `versioneer.get_cmdclass.cmd_version` ([L1900-L1918](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1900-L1918)) - Class
  - `versioneer.get_cmdclass.cmd_version.initialize_options` ([L1905-L1906](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1905-L1906)) - Method
  - `versioneer.get_cmdclass.cmd_version.finalize_options` ([L1908-L1909](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1908-L1909)) - Method
  - `versioneer.get_cmdclass.cmd_version.run` ([L1911-L1918](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1911-L1918)) - Method
  - `versioneer.get_cmdclass.cmd_build_py` ([L1945-L1961](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1945-L1961)) - Class
  - `versioneer.get_cmdclass.cmd_build_py.run` ([L1946-L1961](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1946-L1961)) - Method
  - `versioneer.get_cmdclass.cmd_build_ext` ([L1969-L1993](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1969-L1993)) - Class
  - `versioneer.get_cmdclass.cmd_build_ext.run` ([L1970-L1993](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L1970-L1993)) - Method
  - `versioneer.get_cmdclass.cmd_build_exe` ([L2005-L2024](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2005-L2024)) - Class
  - `versioneer.get_cmdclass.cmd_build_exe.run` ([L2006-L2024](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2006-L2024)) - Method
  - `versioneer.get_cmdclass.cmd_py2exe` ([L2034-L2053](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2034-L2053)) - Class
  - `versioneer.get_cmdclass.cmd_py2exe.run` ([L2035-L2053](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2035-L2053)) - Method
  - `versioneer.get_cmdclass.cmd_egg_info` ([L2062-L2089](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2062-L2089)) - Class
  - `versioneer.get_cmdclass.cmd_egg_info.find_sources` ([L2063-L2089](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2063-L2089)) - Method
  - `versioneer.get_cmdclass.cmd_sdist` ([L2099-L2118](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2099-L2118)) - Class
  - `versioneer.get_cmdclass.cmd_sdist.run` ([L2100-L2106](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2100-L2106)) - Method
  - `versioneer.get_cmdclass.cmd_sdist.make_release_tree` ([L2108-L2118](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2108-L2118)) - Method
  - `versioneer.do_setup` ([L2173-L2227](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2173-L2227)) - Function
  - `versioneer.scan_setup_py` ([L2230-L2264](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2230-L2264)) - Function
  - `versioneer.setup_command` ([L2267-L2271](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingversioneer.py#L2267-L2271)) - Function


### Algorithm Framework & Type System
Defines abstract base classes and primitive types to enforce the Strategy Pattern across the pipeline.


**Related Classes/Methods**:

- `starfish.core.pipeline.algorithmbase.AlgorithmBase`:8-45
- `starfish.core.starfish.starfish`:25-36
- `starfish.core.types._constants.AugmentedEnum`:4-14



**Source Files:**

- [`starfish/core/pipeline/algorithmbase.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/pipeline/algorithmbase.py)
  - `starfish.core.pipeline.algorithmbase.AlgorithmBase` ([L8-L45](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/pipeline/algorithmbase.py#L8-L45)) - Class
  - `starfish.core.pipeline.algorithmbase.AlgorithmBase.__init__` ([L9-L12](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/pipeline/algorithmbase.py#L9-L12)) - Method
  - `starfish.core.pipeline.algorithmbase.AlgorithmBase.run_with_logging` ([L15-L45](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/pipeline/algorithmbase.py#L15-L45)) - Method
  - `starfish.core.pipeline.algorithmbase.AlgorithmBase.run_with_logging.helper` ([L22-L44](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/pipeline/algorithmbase.py#L22-L44)) - Function
- [`starfish/core/starfish.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/starfish.py)
  - `starfish.core.starfish.art_string` ([L11-L20](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/starfish.py#L11-L20)) - Function
  - `starfish.core.starfish.starfish` ([L25-L36](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/starfish.py#L25-L36)) - Function
  - `starfish.core.starfish.version` ([L40-L42](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/starfish.py#L40-L42)) - Function
  - `starfish.core.starfish.util` ([L47-L51](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/starfish.py#L47-L51)) - Function
  - `starfish.core.starfish.install_strict_dependencies` ([L55-L63](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/starfish.py#L55-L63)) - Function
- [`starfish/core/types/_constants.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_constants.py)
  - `starfish.core.types._constants.AugmentedEnum.__hash__` ([L5-L6](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_constants.py#L5-L6)) - Method
  - `starfish.core.types._constants.AugmentedEnum.__eq__` ([L8-L11](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_constants.py#L8-L11)) - Method
  - `starfish.core.types._constants.AugmentedEnum.__str__` ([L13-L14](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_constants.py#L13-L14)) - Method
- [`starfish/core/types/_functionsource.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_functionsource.py)
  - `starfish.core.types._functionsource.FunctionSourceBundle` ([L13-L51](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_functionsource.py#L13-L51)) - Class
  - `starfish.core.types._functionsource.FunctionSource` ([L54-L82](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_functionsource.py#L54-L82)) - Class
  - `starfish.core.types._functionsource.FunctionSource.__init__` ([L71-L73](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_functionsource.py#L71-L73)) - Method
  - `starfish.core.types._functionsource.FunctionSource.__call__` ([L75-L77](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/types/_functionsource.py#L75-L77)) - Method


### Diagnostic & Simulation Engine
Provides tools for framework verification, synthetic data generation, and visualization utilities.


**Related Classes/Methods**:

- `starfish.core.experiment.builder.defaultproviders.RandomNoiseTile`:16-39
- `starfish.util.plot.diagnose_registration`:175-223



**Source Files:**

- [`notebooks/py/BaristaSeq.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/BaristaSeq.py)
  - `notebooks.py.BaristaSeq.plot_scaling_result` ([L259-L277](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/BaristaSeq.py#L259-L277)) - Function
- [`notebooks/py/STARmap.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/STARmap.py)
  - `notebooks.py.STARmap.plot_scaling_result` ([L173-L192](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/STARmap.py#L173-L192)) - Function
- [`notebooks/py/Starfish_simulation.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/Starfish_simulation.py)
  - `notebooks.py.Starfish_simulation.choose` ([L13-L19](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/Starfish_simulation.py#L13-L19)) - Function
  - `notebooks.py.Starfish_simulation.graham_sloane_codes` ([L21-L26](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/Starfish_simulation.py#L21-L26)) - Function
  - `notebooks.py.Starfish_simulation.graham_sloane_codes.code_sum` ([L24-L25](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/Starfish_simulation.py#L24-L25)) - Function
  - `notebooks.py.Starfish_simulation.generate_spot` ([L55-L60](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/Starfish_simulation.py#L55-L60)) - Function
- [`notebooks/py/osmFISH.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/osmFISH.py)
  - `notebooks.py.osmFISH.load_results` ([L131-L133](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/osmFISH.py#L131-L133)) - Function
  - `notebooks.py.osmFISH.get_benchmark_peaks` ([L135-L151](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingnotebooks/py/osmFISH.py#L135-L151)) - Function
- [`starfish/core/experiment/builder/defaultproviders.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py)
  - `starfish.core.experiment.builder.defaultproviders.RandomNoiseTile.shape` ([L22-L23](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L22-L23)) - Method
  - `starfish.core.experiment.builder.defaultproviders.RandomNoiseTile.coordinates` ([L26-L31](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L26-L31)) - Method
  - `starfish.core.experiment.builder.defaultproviders.RandomNoiseTile.format` ([L34-L35](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L34-L35)) - Method
  - `starfish.core.experiment.builder.defaultproviders.RandomNoiseTile.tile_data` ([L37-L39](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L37-L39)) - Method
  - `starfish.core.experiment.builder.defaultproviders.tile_fetcher_factory.ResultingClass.get_tile` ([L87-L95](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/core/experiment/builder/defaultproviders.py#L87-L95)) - Method
- [`starfish/util/plot.py`](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/util/plot.py)
  - `starfish.util.plot.imshow_plane` ([L15-L61](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/util/plot.py#L15-L61)) - Function
  - `starfish.util.plot.intensity_histogram` ([L64-L99](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/util/plot.py#L64-L99)) - Function
  - `starfish.util.plot.overlay_spot_calls` ([L102-L165](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/util/plot.py#L102-L165)) - Function
  - `starfish.util.plot._linear_alpha_cmap` ([L168-L172](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/util/plot.py#L168-L172)) - Function
  - `starfish.util.plot.diagnose_registration` ([L175-L223](https://github.com/CodeBoarding/starfish/blob/master/.codeboardingstarfish/util/plot.py#L175-L223)) - Function




### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)