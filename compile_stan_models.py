# %% original code
#from models.stan_dlm_models import *
# from stan_backend import write_model_files

# Helper to write stan file and either compile via pystan or write a marker for cmdstanpy
# models_to_write = [
#     ('dlm_vanilla_ar1', dlm_vanilla_ar1),
#     ('dlm_vanilla_ar2', dlm_vanilla_ar2),
#     ('dlm_noregs_ar1', dlm_noregs_ar1),
#     ('dlm_dynregs_ar1', dlm_dynregs_ar1),
#     ('dlm_vanilla_ar1_noseasonal', dlm_vanilla_ar1_noseasonal),
#     ('dlm_vanilla_ar2_noseasonal', dlm_vanilla_ar2_noseasonal),
# ]
# write_model_files(models_to_write)

from stan_backend import write_model_files, HAVE_PYSTAN, HAVE_CMDSTANPY
# import stan_backend
import inspect

def get_models_to_write(module, stanversion):
    models_to_write = []
    for name, value in inspect.getmembers(module):
        # Exclude system/private attributes (like __name__, __file__)
        if name.startswith("__") and name.endswith("__"):
            continue
        models_to_write.append( (name, value, stanversion) )
    return models_to_write
    


# %% Now lets compile
if HAVE_PYSTAN:
    print('Pystan2 detected. Compiling pystan models')
    # %% Do this on OSX to get pystan2 to compile - likely unnecessary on other platforms?!!?
    import sys
    if sys.platform == "darwin":
        import subprocess
        import os
        import pystan
        
        # 1. Locate the broken PyStan math file. This error is due to Apple's compiler being strict and crashing when it sees the "trt.derived()" typo
        pystan_dir = os.path.dirname(pystan.__file__)
        buggy_file = os.path.join(pystan_dir, "stan", "lib", "stan_math", "lib", "eigen_3.3.3", "Eigen", "src", "Core", "Transpositions.h")
        
        # 2. Open the file and fix the typo (trt.derived() -> trt)
        with open(buggy_file, "r") as f:
            content = f.read()
        
        if "trt.derived()" in content:
            print("Found Eigen 3.3.3 bug. Patching file...")
            content = content.replace("trt.derived()", "trt")
            with open(buggy_file, "w") as f:
                f.write(content)
            print("Patch applied successfully!\n")
        else:
            print("Eigen 3.3.3 bug is already patched.\n")

        # 3. Setup the compiler environment we successfully tested (MAC only!)
        sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path']).decode('utf-8').strip()
        cpp_include = os.path.join(sdk_path, "usr", "include", "c++", "v1")

        os.environ["CC"] = "/usr/bin/clang++"
        os.environ["CXX"] = "/usr/bin/clang++"
        os.environ["MACOSX_DEPLOYMENT_TARGET"] = "15.0"

        cflags = f"-isysroot {sdk_path} -I{cpp_include} -stdlib=libc++ -std=c++14 -O3"
        os.environ["CFLAGS"] = cflags
        os.environ["CXXFLAGS"] = cflags
    # %% now compile
    import models.pystan2_dlm_models as stanmodels
    models_to_write = get_models_to_write(stanmodels, 'pystan2')
    write_model_files(models_to_write)
    
if HAVE_CMDSTANPY:
    print('cmdstanpy detected. Compiling cmdstan models')
    import models.cmdstanpy_dlm_models as stanmodels
    models_to_write = get_models_to_write(stanmodels, 'cmdstanpy')
    write_model_files(models_to_write)