***CODE University - Intro to Machine Learning Workshop***

Machine Learning isn't a new concept, but it has gained a lot of notoriety in the past few years, particularly as the quantity of available data explodes and the processing power of computers drastically accelerates. You may be hearing the phrase all over the place. "Let's add some ML to this product." "This new feature is powered by Machine Learning." "We just need to get the ML working." The more you hear the phrase, the less it seems like you know what it means.

The purpose of this workshop is to demystify the concept of ML. You will get a chance to learn about the idea as a whole, while also getting hands-on experience with some of the large topics in Machine Learning, such as Supervised Learning and Unsupervised Learning.

You will be able to explore the Titanic Dataset, build out your own Machine Learning models (after we teach you what that means), train and test your models, and analyze the results.

If you have basic Python experience, curiosity, and excitement, you're ready to get started and learn.

### Workshop structure
* Presentation (Intro to ML): 45 minutes
* Hands on experimentation: 3 hours
* Wrap-up and questions: 15 minutes

---

### Getting set up

#### Installing and Configuring Python
* You will need Python 3.* installed on your computer. Go to [the Python website](https://www.python.org/downloads) and find the right download for your OS.

#### Download Anaconda
* Anaconda is a python distribution which includes pretty much everything you need for out-of-the-box data science work.
* Download and install Anaconda (for a Python 3.* version) from [https://www.anaconda.com/download](https://www.anaconda.com/download)

#### Getting the Extra Libraries
* There are a few extra packages you'll need for this workshop. Run the lines below to get everything set up:

```
conda create -n ml-workshop python==3.4 pandas numpy scikit-learn scipy matplotlib seaborn ipython jupyter

source activate ml-workshop
```

#### Downloading the code
You can download the repository zip file or clone the repository.

#### Running the notebooks
* Run the following from the project's root folder:
`jupyter notebook`
* Then, open the `ML Workshop Part 1 - Data Exploration.ipynb` file.
* That's all you need. You're ready to go learn more about Machine Learning.

---

### Resources
* [RMS Titanic History](https://en.wikipedia.org/wiki/RMS_Titanic)
* [Titanic Dataset - History/Info](http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.html)
* [Titanic Dataset/Competition - Kaggle](https://www.kaggle.com/c/titanic)

### More Info about the Libraries
* Pandas - Provides data structures and data analysis tools
* SciPy - Ecosystem of libraries useful for math, science, engineering needs
* Numpy - Scientific computing tools
* Jupyter - Allows creation, sharing, and accessing of documents containing live code, text, etc.
* Seaborn - Data visualization
* Scikit-learn - Tools for data mining and machine learning
* iPython - Used for Python programming in Jupyter notebooks
