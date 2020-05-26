# Basic AI Questions
This is a compiled overview and an attempt to answer all of the quesations in the following [sample sheet](https://datsoftlyngby.github.io/soft2020spring/resources/d67f51f7-AIBasicQuestions.pdf).

## Overview
<ol>

</ol>

## What does AI stand for? [1]
> What	does	AI	stand	for?	Give	you	own	explanation	of	the	meaning	of	it. What is	known as Turing test?

AI Stand for Artificial Intelligence.

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
![what_is_an_agent](./what_is_an_agent.png)


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

