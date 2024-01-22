from threading import Thread
import main
import workloads


def run():
    main.uvicorn.run(main.app, host="172.28.0.12", port=80)


server = Thread(target=run, name="webserver")
wl = Thread(target=workloads.main)
server.join()
