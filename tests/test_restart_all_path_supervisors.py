import time
import myrmidon

def test_path_supervisor_restart_all():
    rt = myrmidon.Runtime()

    # spawn two observed actors under distinct child paths but same prefix
    pid1 = rt.spawn_with_path_observed(10, "/svc/restart/all/one")
    pid2 = rt.spawn_with_path_observed(10, "/svc/restart/all/two")

    # Create the path-scoped supervisor
    rt.create_path_supervisor("/svc/restart/all")

    # Factories to spawn new observed actors during restart
    def f1():
        return rt.spawn_with_path_observed(10, "/svc/restart/all/one")

    def f2():
        return rt.spawn_with_path_observed(10, "/svc/restart/all/two")

    # Register children with the 'restartall' strategy
    rt.path_supervise_with_factory("/svc/restart/all", pid1, f1, "restartall")
    rt.path_supervise_with_factory("/svc/restart/all", pid2, f2, "restartall")

    # stop one child — RestartAll should restart both
    rt.stop(pid1)

    # wait and poll for restart activity
    ok = False
    for _ in range(40):
        children = rt.path_supervisor_children("/svc/restart/all")
        # Check that we have 2 children and at least one has a new PID
        if len(children) == 2 and ((pid1 not in children) or (pid2 not in children)):
            ok = True
            break
        time.sleep(0.05)

    assert ok, "supervisor did not restart children within timeout"
    print("Test Success: Path supervisor 'restartall' strategy verified.")

if __name__ == "__main__":
    test_path_supervisor_restart_all()
