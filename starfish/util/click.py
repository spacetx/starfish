import click


class RequiredParentOption(click.Option):
    """
    For any required option in a parent group, use this type
    so that --help works for the parent command as well as the
    sub-command.
    """

    def handle_parse_result(self, ctx, opts, args):
        # check to see if there is a --help on the command line
        if any(arg in ctx.help_option_names for arg in args):
            # if asking for help see if we are a subcommand name
            for arg in args:
                if arg in ctx.command.commands:
                    # this matches a sub command name, and --help is
                    # present, let's assume the user wants help for the
                    # subcommand
                    cmd = ctx.command.commands[arg]
                    with click.Context(cmd) as sub_ctx:
                        click.echo(cmd.get_help(sub_ctx))
                        sub_ctx.exit()

        return super(RequiredParentOption, self).handle_parse_result(
            ctx, opts, args)
