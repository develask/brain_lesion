import os
import errno    
import os




def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

mkdir_p("../data")

wanted_dirs = ['FA', 'flair', 'L1', 'L2', 'L3', 
			    'MD', 'MO', 'S0', 'V1', 'V2', 'anatomica']

for d in wanted_dirs:
	mkdir_p("../data/"+d)

# for elem in os.walk("."):
# 	direcs= elem
# 	break

# dirs = direcs[1]

for d in wanted_dirs:
		mkdir_p("../data/"+d+"/normalized")
		mkdir_p("../data/"+d+"/div_2D"+"/x")
		mkdir_p("../data/"+d+"/div_2D"+"/y")
		mkdir_p("../data/"+d+"/div_2D"+"/z")
		mkdir_p("../data/"+d+"/slices")

mkdir_p("../data/mask/normalized")
mkdir_p("../data/raw")


	#if direc != datos or 
	#os.mkdirs(direc)




