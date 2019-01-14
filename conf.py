
obj_name = "default"
SHORT_SCAN = False
conf_dic = {
    "project_video":{"SHORT_SCAN" : False},
    "harder_challenge_video":{"SHORT_SCAN" : True},
    "challenge_video":{"SHORT_SCAN" : True},
    "default": {"SHORT_SCAN": False},
}

def load_config(obj_name=obj_name):
    obj = "default"
    if obj_name in conf_dic:
        obj = obj_name
    for key, value in conf_dic[obj].items():
        #exec("global %s" %(key))
        #exec("%s = %d" % (key, value))
        #print(key, value)
        globals()[key] = value

