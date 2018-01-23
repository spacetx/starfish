from ..util.argparse import FsExistsType


class Registration(object):
    @classmethod
    def add_to_parser(cls, subparsers):
        register_group = subparsers.add_parser("register")
        register_group.add_argument("in_json", type=FsExistsType())
        register_group.add_argument("out_dir", type=FsExistsType())
        register_group.set_defaults(starfish_command=Registration.register)
        registration_subparsers = register_group.add_subparsers(dest="registration_algorithm")

        fourier_shift_group = registration_subparsers.add_parser("fourier_shift")
        fourier_shift_group.add_argument("--u", default=1, type=int, help="Amount of up-sampling")
        fourier_shift_group.set_defaults(registration_algorithm=Registration.fourier_shift)

        cls.register_group = register_group

    @classmethod
    def register(cls, args):
        if args.registration_algorithm is None:
            cls.register_group.print_help()
            cls.register_group.exit(status=2)

        args.registration_algorithm(args)

    @classmethod
    def fourier_shift(cls, args):
        import numpy as np

        from ..io import Stack
        from ..register import compute_shift, shift_im

        print('Registering ...')
        s = Stack()
        s.read(args.in_json)

        mp = s.max_proj('ch')
        res = np.zeros(s.shape)

        for h in range(s.num_hybs):
            # compute shift between maximum projection (across channels) and dots, for each hyb round
            shift, error = compute_shift(mp[h, :, :], s.aux_dict['dots'], args.u)
            print("For hyb: {}, Shift: {}, Error: {}".format(h, shift, error))

            for c in range(s.num_chs):
                # apply shift to all channels and hyb rounds
                res[h, c, :] = shift_im(s.data[h, c, :], shift)

        s.set_stack(res)

        s.write(args.out_dir)
