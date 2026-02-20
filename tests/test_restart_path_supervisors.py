import time
import myrmidon


def test_path_supervisor_restart_one():
    rt = myrmidon.Runtime()

    pid = rt.spawn_with_path_observed(10, "/svc/restart/one")
    rt.create_path_supervisor("/svc/restart")

    def factory():
        # factory should create a new observed actor under the same path
        return rt.spawn_with_path_observed(10, "/svc/restart/one")

    # Attach factory-based supervision (expected API)
    rt.path_supervise_with_factory("/svc/restart", pid, factory, "restartone")

    # stop original
    rt.stop(pid)

    # give supervisor a moment to restart
    time.sleep(0.2)

    children = rt.path_supervisor_children("/svc/restart")
    assert len(children) >= 1
