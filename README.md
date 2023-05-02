Download Link: https://assignmentchef.com/product/solved-cmu11485-homework-2-part-2-face-verification-using-convolutional-neural-networks
<br>
Face recognition can be categorized into face classification and face verification. Given an image of a person’s face, the task of classifying the ID of the face is known as face classification, which is a closed-set problem. The task of determining whether two face images are of the same person is known as face verification, which is an open-set problem<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>.

In this assignment, you will use Convolutional Neural Networks (CNNs) to design an end-to-end system for <strong>face verification</strong>. Your system will be given two images as input and will output a score that quantifies the similarity between the <em>faces </em>in these images. This helps us decide whether the faces from the two images are of the same person or not.

You will train your model on a dataset with a few thousand images of labelled ID’s (i.e., a set of images, each labeled by an ID that uniquely identifies the person). You will learn more about embeddings (in this case, embeddings for face information), several loss functions, and, of course, convolutional layers as effective shift-invariant feature extractors. You will also develop skills necessary for processing and training neural networks with big data, which is often the scale at which deep neural networks demonstrate excellent performance in practice.

<h1>2           Face Verification</h1>

The input to your system will be a <em>trial</em>, i.e., a pair of face images that may or may not belong to the same person. Given a trial, your goal is to output a numeric score that quantifies how similar the faces in the two images are. One straightforward approach is to flatten each image matrix into a vector and then to compute the Euclidean distance between two vectors. A lower distance will indicate higher confidence that the faces in the two images are of the same person. If you get a competitive result on Kaggle by applying this approach(you can also define your own distance metric functions), you can skip the following texts except for <em>Dataset </em>and <em>Submission </em>sections because you have finished hw2p2. If you do not get a desirable result or if you don’t want to hurt your CNN’s feelings, the following instructions might help you out.

<h2>2.1         Face Embedding</h2>

We might not really encourage directly computing the distance between two image matrices for two reasons. First, flattened image vectors are typically high-dimensional, which results in additional computation costs. Second, original image features are not discriminative enough. Your task in this assignment is to train a CNN model to extract a compact, low-dimensional feature, which keeps the most important information of the image and is also discriminative. This compact feature will be represented in a <em>fixed-length </em>vector, known as a <strong>face embedding</strong>. Given two face embeddings, you will use an appropriate metric between the embeddings to produce your similarity scores. Tips on how to choose a proper distance metric will be covered later.

<h2>2.2         Getting Started</h2>

If you have trained your CNN, your end-to-end face verification system will use your CNN as follows – given two images, each image will be passed through the CNN to generate corresponding face embeddings, between which you will compute your similarity score. Your system will output this score. The next question is: <strong>how should you train your CNN to produce high-quality face embeddings?</strong>

There are typically two popular approaches, both of which are able to give SOTA results. Feel free to choose one of them or the combination of these two.

<h3>2.2.1         <em>N</em>-way Classification</h3>

Classification is a good start. Similar to speech classification in the previous assignment, you are able to apply CNNs for face classification. Suppose the labeled dataset contains a total of <em>M </em>images that belong to <em>N </em>different people (here, <em>M &gt; N</em>). Your goal is to train your model on this data so that it produces “good” face embeddings. You can do this by optimizing these embeddings for predicting the face IDs from the images. More concretely, your network will consist of several (convolutional) layers for feature extraction. The input will be (possibly a part of) the image of the face. The output of the <em>last </em>such feature extraction layer is the face embedding. You will pass this face embedding through a linear layer with dimensions embedding dim×num faceids, followed by softmax, to classify the image among the <em>N </em>(i.e., num faceids) people. You can then use cross-entropy loss to optimize your network to predict the correct person for every training image. After the network is trained, you will remove the linear/classification layer. This leaves you with a CNN that computes face embeddings given arbitrary face images.

A high testing classification accuracy will <strong>probably </strong>indicate that your feature extractor is good enough to generate discriminative face embeddings. You are encouraged to explore the interconnection between classification accuracy and verification performance.

Though a good job in classification is guaranteed to help you reach the A-cutoff, you are free to apply advanced loss functions such as Center-loss [1], LM [2], L-GM [3], SphereFace [4], CosFace [5], ArcFace [6], UniformFace [7] to go beyond this.

<h3>2.2.2         Metric Learning</h3>

The second approach is actually called deep metric learning(DML): Instead of modeling the classes, you are directly modeling the similarity between two images. The general goal is to make the minimum distance between negative pairs larger than the maximum distance between positive pairs<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>.

A potential approach is to build a Siamese Neural Network [8] and apply a Contrastive loss function as follows:

))]                                                       (1)

Where <em>d </em>denotes Euclidean distance, and <em>y </em>= 1<em>/</em>0 indicates the pair <em>P<sub>i </sub></em>is positive/negative respectively. <em>m </em>is a margin. <em>N </em>denotes total number of training objectives.

There are two popular approaches to make pairs for your verification system. One is <strong>offline selection</strong>: pairs are generated before passed through the neural network. Another is <strong>online selection</strong>: pairs are generated in the mini-batch during training. For offline selection, please pay attention to the ratio of #negative pairs to #positive pairs. You are <strong>advised </strong>to set this ratio as 5:5, 6:4, 7:3. For online selection, one straightforward method is to select all pairs within a mini-batch of size B. You can also just select <em>hard</em><a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> pairs within the mini-batch, which is also referred to as <strong>Hard Sample Mining </strong>[9, 10].

Instead of measuring the similarity between pairs, you can also apply Triplet loss [11] or Quadruplet loss [12] to model the similarities among triplets or quadruplets.

If you’re wondering if there exists a Quintuplets, Sextuplets, Septuplets or even Octuplets loss, you can refer to the N-pair Loss [13], Lifted-Structure Loss [14], Softtriplet Loss [15] papers.

It may also be possible for other advanced loss functions such as Pair-Wise Loss [16], Multi-Similarity(MS) [17], Mask Proxy(MP) [18] to give SOTA verification performance.

<h2>2.3         Loading Training Data</h2>

For loading the images, we recommend that you look into the ImageFolder dataset class of PyTorch at <a href="https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder">https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder</a><a href="https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder">.</a> The images in subfolders of classification data are arranged in a way that is compatible with this dataset class. Note that ImageFolder is helpful for both N-way classification, and Metric Learning tasks.

<h2>2.4         System Evaluation</h2>

This subsection briefly describes how the “quality” of your similarity scores will be evaluated. Given similarity scores for many trials, some <em>threshold </em>score is needed to actually accept or reject pairs as <em>same-person </em>pairs (i.e., when the similarity score is above the threshold) or <em>different-person </em>pairs (i.e., when the score is below the threshold), respectively. For any given threshold, there are four conditions on the results: some percentage of the different-person pairs will be accepted (known as the <em>false positive </em>rate), some percentage of the same-person pairs will be rejected (known as the <em>false rejection </em>rate), some percentage of the differentperson pairs will be rejected (known as the <em>true negative </em>rate), and some percentage of the same-person pairs will be accepted (known as the <em>true positive </em>rate).

The Receiver Operating Characteristic (ROC) curve is created by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings <a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a>. The Area Under the Curve (AUC) for the ROC curve is equal to the probability that a classifier will rank a randomly chosen similar pair (images of same people) higher than a randomly chosen dissimilar one (images from two different people) (assuming ’similar’ ranks higher than ’dissimilar’ in terms of similarity scores).

This is the metric which will be used to evaluate the performance of your model for the face verification task.

To track your progress, after an epoch of training, you can compute a similarity score for every trial in the validation set, write them to another file. One suggested approach to compute AUC is to use the function provided in sklearn library<a href="#_ftn5" name="_ftnref5"><sup>[5]</sup></a>:

<ul>

 <li>sklearn.metrics.roc auc score(true label, similarity scores). This function is useful for Verification Validation. It loads the true label array and the generated similarity scores array and prints out the average AUC score. Please also pay attention to the difference between cosine similarity score and Euclidean distance score.</li>

</ul>

<h2>2.5         Cosine Similarity VS Euclidean Distance</h2>

You may struggle with selecting a proper distance metric for the verification task. The most two popular distance metrics used in verification are cosine similarity and Euclidean distance. We would tell you in that both two metrics are able to reach SOTA score, but at least you should get an intuition on how to choose one of them.

The metric should be training-objective-specific, where training objective refers to the loss function. Let us start with revisiting softmax cross entropy:

(2)

Where <em>Y<sub>i </sub></em>is the label of <em>X<sub>i</sub></em>. If you take a thorough look at this formula, you will find that the objective is to make the vector(embedding) <em>X<sub>i </sub></em>be closer to the vector <em>W<sub>Y</sub></em><em><sub>i </sub></em>and be far away from other vectors <em>W<sub>Y</sub></em><em><sub>j</sub></em>. Under this rule, the <em>W<sub>Y</sub></em><em><sub>i </sub></em>is actually the center of <em>i</em>−<em>th </em>class. Because you are performing dot product between the class center and the embedding, then each embedding would be similar to its center in the <strong>Angular Space</strong>, which could be illustrated in the following Figure. 1. So during verification, you are strongly suggested to apply cosine similarity rather than Euclidean distance to compute the similarity score.

Figure 1: Angular Space [4]

Furthermore, if we design our own loss function e.g. in Eq. 3, you are suggested to apply Euclidean distance metric to compute similarity. (Is this RBF?)

(3)

Question left to you, what metric is <strong>probably </strong>better if you start with metric learning and apply the loss function in Eq. 1?

However, the aforementioned conclusions are not definitely true. We would tell you that sometimes Euclidean distance is also good when you apply softmax XE in Eq. 2 and cosine similarity is also good when you apply Eq. 3 as loss function. We would just give you the following hint and let you explore it.

(4)

<h1>3           Dataset</h1>

The data for the assignment can be downloaded from the Kaggle competition link <a href="#_ftn6" name="_ftnref6"><sup>[6]</sup></a>. The dataset contains images of size 64 × 64 for all RGB channels.

<h2>3.1         File Structure</h2>

The structure of the dataset folder is as follows:

<ul>

 <li>classification data: Each sub-folder in train data, val data and test data contains images of one person and the name of that sub-folder represents their ID.

  <ul>

   <li>train data: You are supposed to use the train data set for training your model <strong>either for the classification task or for the metric learning task.</strong></li>

   <li>val data: If you are doing with classification task, you are supposed to use val data to validate the classification accuracy. If you are doing with metric learning task, you can skip this folder.</li>

   <li>test data: If you are doing with classification task, you are supposed to use test data to test the classification accuracy. If you are doing with metric learning task, you can skip this folder.</li>

  </ul></li>

 <li>verificationdata: This is the directory that contains the images for both the Verification Validation and Verification Test.</li>

 <li>verificationpairs val.txt: This file contains the trials for Verification Validation. The first two column are the images path of the trial. The third column contains the true label for the pair. You are supposed to use the data in this file to validate your AUC score.</li>

 <li>verification pairs test.txt:This file contains the trials for Verification Test. The first two column are the images path of the trial. You task is to compute the similarity between each two trials and to generate submission file based on this.</li>

 <li>hw2p2 sample submission.csv: This is a sample submission file.</li>

</ul>


