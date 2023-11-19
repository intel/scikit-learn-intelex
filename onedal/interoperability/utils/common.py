import onedal

HostPolicy = onedal._backend.host_policy
DefaultHostPolicy = onedal._backend.default_host_policy

def check_attr(obj, name: str, checker = None) -> bool:
    if checker is None:
        check = lambda x: True
    else:
        check = checker
    try:
        if hasattr(obj, name):
            attr = getattr(obj, name)
        else:
            attr = obj[name]
        return check(attr)
    except Exception:
        return False
    

def is_host_policy(policy):
    is_host = isinstance(policy, HostPolicy)
    is_default = isinstance(policy, DefaultHostPolicy)
    return is_host or is_default
