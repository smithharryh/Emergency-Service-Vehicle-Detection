<h1> 
Audio Event Detection for Emergency Service Vehicles
</h1>


The aim of this project is to create a siren detection system which can detect an incoming siren in a noisy environment. To realise this goal, I have built a deep learning system to correctly classify sirens. While considerable research is available in deep learning for computer vision, audio classification information is more sparse. To compensate for this, I have researched how to cross pollinate the two techniques; using established deep learning programs for computer vision to solve an audio detection problem. Examples for developing spectrograms and other audio visualisation techniques are available, and these are what will be used to train the deep learning model.

<hr>

<h2>Requirements</h2>
I have added a requirements folder for the packages installed in the conda environment, but a lot of code is system dependent on data location etc. The trained models are all deployable (the streamlit website is an example of this).

<h2>Acknowledgments</h2>
Credit goes to Seth Adams and Mike Smalles for both of their projects in a similar region. Mike's project provided inspiration and guidance for data visulisation and developent of the MLP model. Seth's was useful for code reusability and helped to create an architecture fot the other three models.

