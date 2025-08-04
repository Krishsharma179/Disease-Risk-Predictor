from setuptools import find_packages,setup
HYPEN='-e .'

def read_requirements(file:str):
    """
    This function will take the file as the input and read the requirement

    """
    with open(file,'r') as file_object:
        requirement=file_object.readlines()
        requirement=[req.replace("\n"," ") for req in requirement]
        if HYPEN in requirement:
            requirement.remove(HYPEN)

    return requirement 


setup(
    name="MINI_PROJ_SEM5",
    version="0.0.1",
    author="Krish",
    author_email="krishsharma5272@gamil.com",
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt')
)

