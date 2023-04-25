import importlib
import os
import signal
import sys
import time

from modules import timer
from modules import shared
from modules.shared import cmd_opts


import modules.ui

startup_timer = timer.Timer()

if cmd_opts.server_name:
    server_name = cmd_opts.server_name
else:
    server_name = "0.0.0.0" if cmd_opts.listen else None


def wait_on_server(demo=None):
    while 1:
        time.sleep(0.5)
        if shared.state.need_restart:
            shared.state.need_restart = False
            time.sleep(0.5)
            demo.close()
            time.sleep(0.5)
            break


def initialize():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)
    signal.signal(signal.SIGINT, sigint_handler)


def webui():
    initialize()
    while 1:
        shared.demo = modules.ui.create_ui()
        startup_timer.record("create ui")
        # if cmd_opts.gradio_queue:
        #     shared.demo.queue(64)

        gradio_auth_creds = []
        if cmd_opts.gradio_auth:
            gradio_auth_creds += [x.strip() for x in cmd_opts.gradio_auth.strip('"').replace('\n', '').split(',') if
                                  x.strip()]
        if cmd_opts.gradio_auth_path:
            with open(cmd_opts.gradio_auth_path, 'r', encoding="utf8") as file:
                for line in file.readlines():
                    gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]

        app, local_url, share_url = shared.demo.launch(
            share=cmd_opts.share,
            server_name=server_name,
            server_port=cmd_opts.port,
            ssl_keyfile=cmd_opts.tls_keyfile,
            ssl_certfile=cmd_opts.tls_certfile,
            debug=cmd_opts.gradio_debug,
            auth=[tuple(cred.split(':')) for cred in gradio_auth_creds] if gradio_auth_creds else None,
            inbrowser=cmd_opts.autolaunch,
            prevent_thread_lock=True
        )
        # after initial launch, disable --autolaunch for subsequent restarts
        cmd_opts.autolaunch = False

        startup_timer.record("gradio launch")

        print(f"Startup time: {startup_timer.summary()}.")

        wait_on_server(shared.demo)
        print('Restarting UI...')

        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)