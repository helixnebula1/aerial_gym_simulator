# Print the members of a class.  Useful for finding out what's in env.  I also used it for env_cfg
def all_members(obj, obj2):
    with open('members.txt', 'a') as fp:
        dirobj = str((dir(obj)))
        dirobj2 = "\n" + str((dir(obj2)))
        fp.write(dirobj)
        #print(dir(env._create_envs))
        fp.write(dirobj2)
    return