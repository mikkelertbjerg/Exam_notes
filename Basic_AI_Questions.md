# Basic AI Questions
This is a compiled overview and an attempt to answer all of the quesations in the following [sample sheet](https://datsoftlyngby.github.io/soft2020spring/resources/d67f51f7-AIBasicQuestions.pdf).

## Overview
<ol>

</ol>

## What does AI stand for? [1]
> What	does	AI	stand	for?	Give	you	own	explanation	of	the	meaning	of	it. What is	known as Turing test?

AI stands for [Artificial Intelligenc](https://en.wikipedia.org/wiki/Artificial_intelligence).

As the name implies it's simulating, or emulating something that's intenlligent, but nessecarily isn't in reality.

>Artificial Intelligence (AI) is a “a huge set of tools for making computers behave intelligently” and in an automated fashion. This includes including voice assistants, recommendation systems, and self-driving cars. </br> [Source](https://www.datacamp.com/community/blog/ai-and-ml?utm_source=adwords_ppc&utm_campaignid=9942305733&utm_adgroupid=100189364546&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=332602034352&utm_targetid=aud-299261629574:dsa-929501846124&utm_loc_interest_ms=&utm_loc_physical_ms=9067633&gclid=CjwKCAjw_LL2BRAkEiwAv2Y3SUl0t4sqlwiVQCtwfN7q20dXCc_gvn__wQLHL2MVS4YZZYbab5Wk0xoCb00QAvD_BwE)

[The turing test](https://en.wikipedia.org/wiki/Turing_test) is a test that is developed to determine whether a machine has human like intelligence or not. The test essentially boils down to a human being communicating with a machine, without knowning beforehand whether its a machine or not. With that in mind the human has to determine if he or she is communicating with a machine or a human being. The test subjects, i.e the machine and the human, has to be apart, so the human can't visibly tell whether or not it's a machine or not based on physical appearance.

### What is an Agent? [2]
>What	is	an	intelligent agent	in	the	context	of	AI? Which	are	its	components?

An <b>[agent](https://www.youtube.com/watch?v=XO6SV0Mup1E)</b> is anything that can be viewed as perceiving its environment through sensors and acting upon that enviroment through actuators or motors.

An example would be:
A human has eyes, ears and other organs which are sensors, i.e "tools" that are used to percieve the enviroment.
A human also has hands, legs, arms, which are actuators, i.e "tools" used to act upon what is percieved in the enviroment.

Relating that to sowftware:
Keystrokes, recieving network packets, cameras, bluetooth, etc. would be sensors that it can act upon by displaying content, sending network packets, writing files based on that input.

It can be illustrated the following way:
![what_is_an_agent](./img/what_is_an_agent.png)


## How can AI be implemented in games? [3]
>How	can	AI	be	implemented	in	games?	Name	some	principles,	methods	and	algorithms	
applied	by	AI agents	in	playing	games.

Historically AIs has been applied to games for ever. They work based in more constrained enviroments, for example in turn based games, but modern AIs also excel at much more complex games. There are specially great discoveries made in the MOBA genre.

Chess fits above description very well, and has had AI available for a while. The agent can given the ruleset of the game, make a descission on a legal move. Furthermore an advanced agent will be able to make a "good" move based on its sensors, using it acuators.

## Which AI branches are you familiar with? [4]
>Which	AI	branches you	are	familiar	with? Give	examples	of	AI	application	areas.

Games, text recognition, speech recognition, image recognition, google in general have massive AIs predicting optimal routes in traffic, utilizing google maps, all of the above apply to many indstirues and business sectors. Comercials and SoMe in general also heavily utilizes AI to for target branding.

## What is Machine Learning? [5]
>What	is	machine learning? Give you own	explanation	of	the	meaning	of	it. Compare	it	to deep learning.

Machine learning is the concepts of a machine "learning" from the input its givin. This can be done by "training" a machine. In principle this often means that you give a machine x amount of data, but hold back y amount of data. Use one or more approiate algorithm(s) for a given task, and let the machine learn from the x data, and finally compare the machines results to y data. Then there are several metrics and parameters that can be used to describe whether or not the machine learned anything valueable. If not algorithm or data paramters can be tweaked for further experimentation.

>Machine Learning (ML) is the “field of study that gives computers the ability to learn without being explicitly programmed.” The lion’s share of ML involves computers learning patterns from existing data and applying it to new data in the form of making predictions, such as predicting whether an email is spam or not, whether a customer will churn or not, and diagnosing a particular piece of medical imaging. </br> [Source](https://www.datacamp.com/community/blog/ai-and-ml?utm_source=adwords_ppc&utm_campaignid=9942305733&utm_adgroupid=100189364546&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=332602034352&utm_targetid=aud-299261629574:dsa-929501846124&utm_loc_interest_ms=&utm_loc_physical_ms=9067633&gclid=CjwKCAjw_LL2BRAkEiwAv2Y3SUl0t4sqlwiVQCtwfN7q20dXCc_gvn__wQLHL2MVS4YZZYbab5Wk0xoCb00QAvD_BwE)

[Deep learning](https://www.datacamp.com/community/blog/ai-and-ml?utm_source=adwords_ppc&utm_campaignid=9942305733&utm_adgroupid=100189364546&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=332602034352&utm_targetid=aud-299261629574:dsa-929501846124&utm_loc_interest_ms=&utm_loc_physical_ms=9067633&gclid=CjwKCAjw_LL2BRAkEiwAv2Y3SUl0t4sqlwiVQCtwfN7q20dXCc_gvn__wQLHL2MVS4YZZYbab5Wk0xoCb00QAvD_BwE) is a form of machine learning that uses algorithms called [neural networks](https://www.youtube.com/watch?v=aircAruvnKk), which are loosely inspired by biological neural networks in human brains.

## Tasks solved by machine learning [6]
>Which	are	the	basic	types	of	tasks	solved	by machine learning?	How	do	they	differ?

Seeing as ML (is a branch of AI), ML can solve many tasks that AI pracially also can solve, or rather AI would solve those tasks through ML. Typically, ML is used for basic predictions based on the data the algorithms has been fed. Comon tasks include text recognition and statiscal analysis. Most of ML tasks also rely on categorical or numerical data.

## Describe the process of ML [7]
>Describe	the	process	of	machine learning.	Which	activities	would	you	plan	to	solve	a	task	
by	implementing	machine learning methods? Draw	a	simple	sequence	diagram.

<ol>
    <li>Find a dataset</li>
    Find a dataset that would be interesting to make predictions about. This could be about more or less anything. However, the more data that is available the higher success rate we'll eventually see.
    <li>'Chalange' the data</li>
    Ask questions about the data. What can this particular dataset help us answer. Which answers could be extracted from the dataset? Which hypothesis can be made?
    <li>Preprocessing</li>
    Preprocess the data so it fits the tasks at hand and don't have any contaminated values. Furthermore, make the data 'computer readable'.
    <li>Analyze</li>
    Analyze which models would be suitable to train the given data on. This might be a given earlier on in the process as many AI/ML tasks are solved in similar ways, based on the data that is being processed.
    <li>Fit & Train</li>
    Fit and train the selected models with the preprocessed data.
    <li>Adjust</li>
    Fiddle with the numbers, and model inputs, see if the results can be tweaked for the better. And if for the worse, ask questions about why? This can be taken back to the preprocessing stage, or maybe we're asking the wrong questions?
    <li>Results</li>
    Analyze the results, and ask critical questions about why certain results were achievede. Were the results in reality too good, and if so, ask why? 
</ol>

I've attempted to illustrate this process with a sequence diagram. That's to say, we don't nesscecarily go through all the steps in the diagram in this particular sequence. But it's ment to fit above described flow
![ML_process](./img/ML_process.svg)

## Supervised and unsupervised learning [8]
>What	is	the	difference	between	supervised	and	unsupervised	machine learning? Give	an	
example	from	the	everyday	life.

The two terms are sort of implied, [supervised learning is learning that is being monitored, whereas unsupervised learning](https://www.youtube.com/watch?v=cfj6yaYE86U) is not.

It's important to note that the data for supervised learning needs to be labeled. mainly there are two types of supervised learning: classification and regression.

Unsupervised learning the data is not labeled, and we don't nessecarily know much about the data beforehand. There are generally fewer algorithms available for unsupervised learning. Clustering is commonly used.

## Which data structures are used for ML? [9]
>Which	data	structures	are	used	to	hold	the	data	needed	for	machine learning?

Some comon structures include: [tuples](https://en.wikipedia.org/wiki/Tuple), [tectors](https://en.wikipedia.org/wiki/Feature_(machine_learning)) or simply put, one-dimensional [array](https://en.wikipedia.org/wiki/Array_data_structure), [matrixs](https://en.wikipedia.org/wiki/Matrix_(mathematics)), [tensors](https://en.wikipedia.org/wiki/Tensor).

## What is Gaussian distribution? [10]
>What	is	the	shape	of	Gaussian	distribution	of	a	set	of	data	values?	What	information	does	
it	provide? Which	parameters	of	a	data	set	are	important	for	its	description?

[The Gaussian distribution](https://www.youtube.com/watch?v=rzFX5NWojp0), also refered to as '[normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)' and 'the bell curve'. As implied by its alternate names the shape of the gaussian distribution is a bell. On the y-axsis the frequency of something represented, in other words, the farther up on the y-axsis, the more frequently that datapoint appears. On the x-axsis the outliers of the dataset are clearly represented. The lowest value is represented (0,0), whilst the average value will be dead-center on the x-axis (.5, x) and the highest value at the end of the x-axis (1, 0). The width of the curve is define by the [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation).

In order to draw a normal distribution the following data is needed:
<ol>
    <li>The average measurement</li>
    <li>The standard deviation of the measurement</li>
</ol>

![Normal distribution](./img/Normal_Distribution.svg)

## In ML what is a feature and a label [11]
>In	machine learning what	is	a	feature	and	what	is	a	label?	Illustrate	with	appropriate	example

Labels can be described as something that categories data.
Features as the name implies, is features of said data.
In combinations [features of some data is how we determine to label data](https://www.youtube.com/watch?v=rnv0-lG9yKU).

An example: A movie with genre g, and run time r, actors a1, a2, a3, a4, a5 and so on, can be labeled as something I personally like or dislike based on the given features (g, r, a1 and so on).

## Lack of data [12]
>How	would	you	proceed,	if	you	do	not	have	sufficient	data	for	building	a	reliable	model?

There are quite a few options:
One could look for a new and more sufficient dataset.
It would also be possible to generate more 'fake' data yourself, there are algorithms available to do so.
