# case-study-duqinaerfa
## project I choose: Scikit-learn
## Technology and Platform used for development
a.What coding languages are used? Do you think the same languages would be used if the project was started today? What languages would you use for the project if starting it today?   
Answer: Python and very little amount of cython. If it was started today, I think python would also be used and I would also use python as python is the most powerful language in AI&ML. Python also has many libraries which makes it more convenient.  
b.What build system is used (e.g. Bazel, CMake, Meson)? What build tools / environment are needed to build (e.g. does it require Visual Studio or just GCC or ?)  
Answer: Version 0.4 made an improvement of the build system to (optionally) link with MKL. Also, provide a lite BLAS implementation in case no system-wide BLAS is found. GCC.  
c.What frameworks / libraries are used in the project? At least one of these projects don’t use any external libraries or explicit threading, yet is noted for being the fastest in its category--in that case, what intrinsic language techniques is it using to get this speed.  
Answer: Scipy.Most of scikit-learn assumes data is in NumPy arrays or SciPy sparse matrices of a single numeric dtype. These do not explicitly represent categorical variables at present. Thus, unlike R’s data.frames or pandas.DataFrame, we require explicit conversion of categorical features to numeric values.  
## Testing: describe unit/integration/module tests and the test framework
a.How are they ensuring the testing is meaningful? Do they have code coverage metrics for example?  
Answer: Use codecov to test the code coverage metrics as it has a /.codecov.yml file.  
b.What CI platform(s) are they using (e.g. Travis-CI, AppVeyor)?  
Answer: Travis-CI as it has a  /.travis.yml file.  
c.What computing platform combinations are tested on their CI? E.g. Windows 10, Cygwin, Linux, Mac, GCC, Clang  
Answer: Linux. # Linux environment to test scikit-learn against numpy and scipy master  
## Software architecture in your own words, including:
a.How would you add / edit functionality if you wanted to? How would one use this project from external projects, or is it only usable as a standalone program?   
When adding additional functionality, provide at least one example script in the examples/ folder. Have
a look at other examples for reference. Examples should demonstrate why the new functionality is useful in
practice and, if possible, compare it to other methods available in scikit-learn. It can load from external datasets. scikit-learn works on any numeric data stored as numpy arrays or scipy sparse matrices. Other types that are convertible to numeric arrays such as pandas DataFrame are also acceptable.
Here are some recommended ways to load standard columnar data into a format usable by scikit-learn:
• pandas.io provides tools to read data from common formats including CSV, Excel, JSON and SQL. DataFrames
may also be constructed from lists of tuples or dicts. Pandas handles heterogeneous data smoothly and provides tools for manipulation and conversion into a numeric array suitable for scikit-learn.  
• scipy.io specializes in binary formats often used in scientific computing context such as .mat and .arff  
• numpy/routines.io for standard loading of columnar data into numpy arrays  
• scikit-learn’s datasets.load_svmlight_file for the svmlight or libSVM sparse format  
• scikit-learn’s datasets.load_files for directories of text files where the name of each directory is the
name of each category and each file inside of each directory corresponds to one sample from that category
For some miscellaneous data such as images, videos, and audio, you may wish to refer to:  
• skimage.io or Imageio for loading images and videos into numpy arrays  
• scipy.io.wavfile.read for reading WAV files into a numpy array  
Categorical (or nominal) features stored as strings (common in pandas DataFrames) will need converting to
numerical features using sklearn.preprocessing.OneHotEncoder or sklearn.preprocessing. OrdinalEncoder or similar. See Preprocessing data.  
b.What parts of the software are asynchronous (if any)?  
Answer: null   
c.Please make diagrams as appropriate for your explanation   
  
d.How are separation of concerns and information hiding handled?  
Answer: In the github file everything is well explained in readme and is well separated in different file packages.  
e.What architectural patterns are used   
Answer: An object in object-oriented programming language such as Python requires building blocks that consist of multiple chunks of code portions that require composition as a step-by-step procedure. The object can reach the overall complete status, once all the blocks are built for the object. Builder design pattern aids in decoupling the construction of a complex object from its mere representation.  
f.Does the project lean more towards object oriented or functional components  
Answer: In six functional components: Classification, regression, clustering, data dimensionality reduction, model selection and data preprocessing.   
## Analyze two defects in the project--e.g. open GitHub issue, support request tickets or feature request for the project
a.Does the issue require an architecture change, or is it just adding a new function or?  
Answer: most issues just adding a new function rather than architecture change.  
b.make a patch / pull request for the project to fix problem / add feature  
Answer；It seems you can not make a patch / pull request for the project to fix problem / add feature if you are not one of the team members.  
## Making a demonstration application of the system, your own application showing how the software is used  
