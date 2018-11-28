import click


class RequiredParentOption(click.Option):
    """
    For any required option in a parent group, use this type
    so that --help works for the parent command as well as the
    sub-command.
    """

    def handle_parse_result(self, ctx, opts, args):
        # this is only a group if it has a "commands" field
        if not hasattr(ctx.command, "commands"):
            return
        # check to see if there is a --help on the command line
        if any(arg in ctx.help_option_names for arg in args):
            # if asking for help see if we are a subcommand name
            # but descend as deeply as possible
            for arg in args:
                if arg in ctx.command.commands:
                    # this matches a sub command name, and --help is
                    # present, let's assume the user wants help for the
                    # subcommand or a subsubcommand
                    cmd = ctx.command.commands[arg]
                    with click.Context(cmd) as sub_ctx:
                        # The following may exit
                        if not self.handle_parse_result(sub_ctx, opts, args):
                            help = cmd.get_help(sub_ctx)
                            # Workaround to include the command in the help
                            name = cmd.name
                            parent = ctx.command.name
                            if name == "starfish":
                                fix = "Usage: "
                            elif parent == "starfish":
                                fix = "Usage: %s %s" % (parent, cmd.name)
                            else:
                                fix = "Usage: starfish %s %s" % (parent, name)
                            help = help.replace("Usage: ", fix)
                            click.echo(help)
                            sub_ctx.exit()

        return super(RequiredParentOption, self).handle_parse_result(
            ctx, opts, args)


def option(*args, **kwargs):
    kwargs["cls"] = RequiredParentOption
    return click.option(*args, **kwargs)
