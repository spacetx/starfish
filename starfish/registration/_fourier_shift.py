from ._base import RegistrationAlgorithmBase


class FourierShiftRegistration(RegistrationAlgorithmBase):
    def __init__(self, upsampling):
        self.upsampling = upsampling

    @classmethod
    def from_cli_args(cls, args):
        return FourierShiftRegistration(args.u)

    @classmethod
    def add_to_parser(cls, subparsers):
        fourier_shift_group = subparsers.add_parser("fourier_shift")
        fourier_shift_group.add_argument("--u", default=1, type=int, help="Amount of up-sampling")
        fourier_shift_group.set_defaults(registration_algorithm_class=FourierShiftRegistration)

    def register(self, stack):
        import numpy as np

        from ..register import compute_shift, shift_im

        mp = stack.max_proj('ch')
        res = np.zeros(stack.shape)

        for h in range(stack.num_hybs):
            # compute shift between maximum projection (across channels) and dots, for each hyb round
            shift, error = compute_shift(mp[h, :, :], stack.aux_dict['dots'], self.upsampling)
            print("For hyb: {}, Shift: {}, Error: {}".format(h, shift, error))

            for c in range(stack.num_chs):
                # apply shift to all channels and hyb rounds
                res[h, c, :] = shift_im(stack.data[h, c, :], shift)

        stack.set_stack(res)

        return stack
