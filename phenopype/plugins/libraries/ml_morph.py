#%% imports

clean_namespace = dir()



#%% namespace cleanup

funs = ['attach', 'ProjectWrapper']

def __dir__():
    return clean_namespace + funs


#%% functions


def attach(proj):
    return ProjectWrapper(proj)
    
class ProjectWrapper: 
    def __init__(self, proj):
        self.project = proj
            
    
    def collect_training_data(self, mode="link"):
        pass        