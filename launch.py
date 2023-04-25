import os
import platform
import shlex
import sys

skip_install = False


def check_python_version():
    is_windows = platform.system() == "Windows"
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro

    if is_windows:
        supported_minors = [10]
    else:
        supported_minors = [7, 8, 9, 10, 11]

    if not (major == 3 and minor in supported_minors):
        import modules.errors

        modules.errors.print_error_explanation(f"""
INCOMPATIBLE PYTHON VERSION

This program is tested with 3.10.6 Python, but you have {major}.{minor}.{micro}.
If you encounter an error with "RuntimeError: Couldn't install torch." message,
or any other error regarding unsuccessful package (library) installation,
please downgrade (or upgrade) to the latest version of 3.10 Python
and delete current Python and "venv" folder in WebUI's directory.

You can download 3.10 Python from here: https://www.python.org/downloads/release/python-3109/

{"Alternatively, use a binary release of WebUI: https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases" if is_windows else ""}

Use --skip-python-version-check to suppress this warning.
""")


def extract_arg(args, name):
    return [x for x in args if x != name], name in args


def prepare_environment():
    global skip_install
    commandline_args = os.environ.get('COMMANDLINE_ARGS', "")

    sys.argv += shlex.split(commandline_args)

    sys.argv, _ = extract_arg(sys.argv, '-f')
    sys.argv, skip_python_version_check = extract_arg(sys.argv, '--skip-python-version-check')
    sys.argv, skip_install = extract_arg(sys.argv, '--skip-install')

    if not skip_python_version_check:
        check_python_version()

    print(f"Python {sys.version}")

    if "--exit" in sys.argv:
        print("Exiting because of --exit argument")
        exit(0)


def start():
    import webui
    webui.webui()


if __name__ == "__main__":
    prepare_environment()
    start()
