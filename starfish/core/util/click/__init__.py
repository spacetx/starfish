from click import (
    argument,
    Choice,
    command,
    Context,
    echo,
    Group,
    group,
    Option,
    ParamType,
    pass_context,
    Path,
)
from click import option as _click_option


class RequiredParentOption(Option):
    """
    For any required option in a parent group, use this type so that --help works
    for the parent command as well as the sub-command.
    """

    def handle_parse_result(self, ctx, opts, args, depth=0):
        # this is only a group if it has a "commands" field
        if hasattr(ctx.command, "commands"):
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
                        with Context(cmd) as sub_ctx:
                            # The following may exit
                            if not self.handle_parse_result(sub_ctx, opts, args, depth + 1):
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
                                echo(help)
                                sub_ctx.exit()
        if depth == 0:
            return super(RequiredParentOption, self).handle_parse_result(
                ctx, opts, args)


def option(*args, **kwargs):
    kwargs["cls"] = RequiredParentOption
    return _click_option(*args, **kwargs)
