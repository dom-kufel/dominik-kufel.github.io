---
layout: post
title: Symmetries, neural nets and applications to quantum - a simple introduction
description: >
  Introduction to symmetries, equivariant neural nets and their limitations. 
sitemap: false
---

<!-- 2023-01-19 -->

<!-- related_posts:
  - /blog/_posts/2023-07-04-toric_code-lre.md -->

<!-- image: /assets/img/blog/summary.png -->


*[cat-cartoon]: It's cute isn't it?
*[unconceal]: Heidegger would be proud haha
 
[^1]: See e.g., [this](https://www.science.org/doi/10.1126/science.add9115). 
[^2]: This once amused me: I asked a top technical exec at a leading LLM company about their view on geometric machine learning (including research on symmetric neural nets), and their first response was: “What’s that?”. 
[^3]: For instance, Hermann Weyl, a famous XX-century mathematical physicist once said "As far as I can see, all a priori statements in physics have their origin in symmetry". 
[^4]: A simple and perhaps naive example is the following: if you want to tell apart cats and dogs you would often want to extract geometrically located features such as whiskers (or their lack) or shape of the face. You would often then want to combine these features in a hierarchical fashion to produce the final label. By the way, condition on the data to successfully apply *local* kernels can be made more formally laid down by considering *quantum enetanglement* of the data: see [this interesting work](https://arxiv.org/abs/2303.11249)!
[^5]: If you have not studied group theory I highly recommend [these excellent lecture notes!](https://people.math.harvard.edu/~ctm/home/text/class/harvard/101/22/html/index.html)
[^6]: On the other hand, global pooling operations at the end of the CNN such as ResNet50 would still keep that layer to be translationally invariant (see e.g., [this](https://maurice-weiler.gitlab.io/cnn_book/EquivariantAndCoordinateIndependentCNNs.pdf#subsection.3.2.5) for a simple proof). Oh, and skip connections are also equivariant, which is not hard to show. 
[^7]: TL;DR: this effect can be attributed to image boundary effects! 
[^8]: Given an equivariant solution for a weight-shared matrix, locality can be imposed by uniformly setting all connections further than $$k$$ (in a geometric sense in $$n$$ dimensions) to strictly $$0$$ such that the kernel does not act there!
[^9]: Of course, in a relative sense translation by 1 pixel is the same for all linear image sizes $$L$$. In an absolute sense, however, linear size of a single pixel scales like $$1/L$$ and thus larger and larger $$\mathbb{Z}_L \times \mathbb{Z}_L$$ becomes a closer depiction to a continuous reality. 
[^10]: Pun not intended! 
[^11]: In case you have not studied Fourier analysis before a good start is [this 3Blue1Brown video](https://www.youtube.com/watch?v=spUNpyF58BY). 
[^12]: Intuitively you can think of the smoothness of non-linearity and new frequency introduction in the following way: take $$RELU$$ (which has a discontinuity in the derivative at $$x=0$$) and imagine you want to fit the sines and cosines to it around $$x=0$$ (in a Fourier transform). To do it one needs extremely small spatial features implying very high frequency Fourier components. In contrast, smoother non-linearities such as swish introduce a much lower-centered frequency spectrum. 
[^13]: In case this is not obvious: see time shifting property [here](https://en.wikipedia.org/wiki/Fourier_transform#Properties).
[^14]: In fact as [[Zhang 2019]](#references) points out, in the early days of CNNs (1990s) [people](https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf) were already aware of the downsampling issue and thus used aliasing-proof blurred-downsampling–average pooling operations. These approaches were abandoned later (2010s) due to better performance of max pooling which re-introduces sensitivity-to-small-shifts due to aliasing. Curious, how extra knowledge of theories on symmetries would be helpful then!

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_symm_neural_net_symmetries_neural_nets_summary.png" width="1000"/></p>


<!-- <p style="text-align:center;"><img src="/assets/img/blog/blogpost_symm_neural_net_approx_symmetries_aliasing.png" width="1000"/></p>

***
<!-- *There is no doubt - AI has seen some tremendous advances in the last decade or so.* -->


<!-- <p>
  <span style="font-size: 1.1em; line-height: 0.6; font-style: italic;">There is no doubt - AI has seen some tremendous advances in the last decade or so. </span>  
</p> -->

<!-- <style>
  .centered-bold-gray {
    text-align: center;         /* Center the text */
    color: black;               /* Set the text color to black */
    font-weight: bold;          /* Make the text bold */
    background-color: #f0f0f0;  /* Set the background color to light gray */
    padding: 10px;              /* Add padding to create space around the text */
    border-radius: 5px;         /* Optional: round the corners of the background */
    display: inline-block;      /* Ensure the background only covers the text area */
  }
</style> -->

<p>
  <span style="font-size: 1.2em; line-height: 0.6; font-style: italic;"> We awe and admire symmetries since at least prehistoric times. </span>  
</p>
So called Acheulean tools (see Fig. 1 below), crafted by a Homo Erectus since almost 2 million (!) years ago[^1], display a nice reflection symmetry, despite most likely, not adding any practical benefit. 


<p style="text-align:center;"><img src="/assets/img/blog/blogpost_symm_neural_net_acheulean_tools.png" width="600" loading="lazy"/></p>
Fig. 1: Prehistoric Acheulean tools are remarkably symmetric. [Image source](https://en.wikipedia.org/wiki/Acheulean) 
{:.figcaption}


Beyond aesthetics, presence of symmetries signifies that an object “looks the same” under certain operations.  For instance “we know” a square looks the same when rotated by $$90^{\circ}$$. But does a neural network know? For instance, take a neural network used to distinguish an images of cats vs dogs. Now suppose we rotate an image of a cat by $$90^{\circ}$$? Will the prediction of the neural network necessarily stay the same? A cat remains a cat under rotation right? 

The answer is an astounding **NO** for a fully generic neural network architecture (see Fig. 2 for a little cat-cartoon). 

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_symm_neural_net_cat_motivation.png" width="600" loading="lazy"/></p>
Fig. 2: Generic neural nets can change their predictions under symmetry transformations such as image rotations.  
{:.figcaption}

That's peculiar! Let's patch things up by therefore reformulating the question 
<p style="text-align: center; color: black; font-weight: bold;">
  Can prediction accuracy and robustness be improved if neural nets <i>were</i> symmetry-aware?
</p>

Well, the answer depends on the task. 

On one hand, as of early 2025, there is a limited set of applications of such “symmetric neural networks” to e.g., large language models[^2].
On the other hand, incorporation of geometric features such as symmetries for neural networks has contributed to many successful applications of AI for science, such as Nobel-prize winning AlphaFold 2 model for protein folding or neural net models [accurately predicting interatomic potentials]((https://www.nature.com/articles/s41467-022-29939-5)). 

Beyond molecular problems, a very natural place for geometric models to perform well is full-fledged quantum physics which to many is almost synonymous to a field describing “symmetries” in nature[^3]. This, at least naively, makes me hopeful about application of symmetry-aware neural nets to quantum problems, particularly in the many body context, where many of the new symmetries of the system can emerge. Is this hope well founded?  

In the series of two blogposts, I will try to convince you that this is indeed the case. I initially wanted to only write one blogpost on this topic but then realized that there is just so much stuff to talk about! Therefore, in this more **ML-focused blogpost**, I will review symmetries for neural networks from my favorite angle, which I hope is conceptually the cleanest. I do not assume previous knowledge of group theory or physics, only standard linear algebra and basic ML knowledge. We will gently start <a href="#symmetriesneural-networks-from-cnns-to-symmetry-groups-and-back">in the next section</a> from reviewing very basic ideas about symmetries, groups, representations and proceeding to distinguishing two types of symmetric transformations: "invariance" and "equivariance". <a href="#how-to-explicitly-construct-symmetric-neural-nets">Next</a> we will discuss a bit about how to ensure symmetric transformations of neural networks by using two leading methods: data augmentation and equivariant neural nets. <a href="#how-to-explicitly-construct-symmetric-neural-nets">Finally</a>, we will close off with limitations of the both approaches and set the stage for applying these models to quantum many-body physics context. In the second blogpost we will build on this knowledge, take a little bit more of a physics angle on things, and see some surprising connections between some more-or-less obvious approaches to symmetries in physics and their re-discovery in the ML community. Exciting stuff ahead, so let's explore symmetries together! 

* table of contents
{:toc}


## Symmetries&neural networks: from CNNs to symmetry groups and back

Let’s start from discussing 2D images. One of the early revolutions in image recognition was an invention of convolutional neural networks (CNNs): see e.g., [[LeCun+ 1995]](#references). Apart from “going deep” [[Goodfellow+ 2016]](#references), their incredible success may be largely attributed to structural biases they incorporate, reflecting two underlying properties of most images: (i) locality of the information contained (ii) translational symmetry in extracting features. 

For locality, roughly speaking, objects can be recognized by hierarchical combination of geometrically local features[^4]. To motivate translational symmetry, the simplest example is the most cliche ever: if you want to recognize cats vs dogs, it should not matter *where* the cat is on an image: a cat is a cat. This is not ensured in a generic neural network architecture: even if the network has correctly predicted a cat on an image, shifting it by some amount can switch a label to a dog - despite we know this just can’t be right! Of course, this is just a very naive way of looking at things. A framework of **geometric machine learning**, formalizes many of the concepts related to symmetries and inductive biases for neural nets, such that we can study many more exotic symmetries than translations (special to CNNs). Let’s therefore try to be more precise. First, what do we even mean by translational symmetry, or symmetry more generally? 


<!-- (GOOD example for geometrically local features???) -->

### Symmetries and groups

Symmetry of an object is a transformation that leaves it *invariant* (i.e. the object does not change). The mathematical framework to capture symmetries is group theory[^5]. Symmetry transformations make mathematical groups. For instance, one can make combinations of rotations by $$90^{\circ}$$ and flips of a square to make a $$D_4$$ “dihedral” group (see Fig. 3). Mathematically structure of the group implies (i) combination of symmetries gives other symmetries (**closure**) (ii) existence of an **identity operation** (e.g., do not rotate nor flip the square at all!) and (iii) existence of an **inverse element** (e.g., if you want to “undo” rotation by $$90^{\circ}$$, apply rotation by $$270^{\circ}$$ to revert to the original position). Thus for a $$D_4$$ group we can explicitly write down the group elements $$G=\{ e, r, r^2, r^3, f, rf, r^2 f, r^3 f \}$$ where $$e$$ denotes identity operation, $$r$$ represents a clockwise rotation by $$90^{\circ}$$ and $$f$$ a horizontal flip around centered $$y$$-axis. 


<p style="text-align:center;"><img src="/assets/img/blog/blogpost_symm_neural_net_square.png" width="600" loading="lazy"/></p>
Fig. 3: Symmetries of a square.  
{:.figcaption}

Notation comment: sometimes it is convenient to write down the group in the form of the [group presentation](https://en.wikipedia.org/wiki/Presentation_of_a_group) $$G = \langle r,f \vert r^4=f^2=e, rf=f r^{-1} \rangle$$ where we note down **group generators** which, when multiplied by other generators and themselves give all possible group elements ($$r,f$$ are generators for $$D_4$$), together with a set of group relations i.e. constraints between different group elements $$r^4=f^2=e, rf=f r^{-1}$$ above. 

**Exercise:** Convince yourself that $$G$$ has all elements listed above and not less/more! E.g., what operation does $$f r^2$$ correspond to? 
{:.message}

<blockquote class="note">
  <b>Upshot:</b> Symmetries can be naturally described by group theory. 
</blockquote>

### Groups and representations

Before we get back to the neural nets, let's discuss one more concept: an idea of representation of a group. Mathematically, it is a (not neccesarily one-to-one) map $$\rho : G \rightarrow GL(N,\mathbb{R})$$ from group elements to a (real) $$N \times N$$ matrices (known as a general linear group $$GL(N,\mathbb{R})$$). This map is required to be a **homomorphism** i.e. fulfills $$\rho(g_1 g_2) = \rho (g_1) \rho (g_2) \ \forall g_1,g_2 \in G$$. Homomorphism is helpful here since it means that we inherit group multiplication property (listed as closure above) also when multiplying corresponding matrices in the representation. Intuitively, the reason why we introduce representations is just to study groups more concretely i.e. in terms of matrices which can act on some geometric spaces and where we can play around with linear algebra methods we are very used to! 

For every group there are many possible ways of representing it. Let's talk about some notable ones: a representation is **trivial** if $$\rho(g) = \mathbb{1} \ \forall g \in G$$ and is **faithful** if $$\rho$$ is one-to-one (i.e. every group element maps to a unique matrix in the representation). Another relevant representation is a **regular representation** of the group is that where all ($$\vert G \vert$$) group elements are represented as permutations on ($$\vert G \vert$$) elements. 

It's a bit abstract so let's look at a simple example. Consider a $$\mathbb{Z}_d$$ group $$G=\{ e, f, \dots, f^{d-1} \}=\langle f \vert f^d = e \rangle$$. A trivial representation is just $$\rho(e)=\rho(f)=\dots=\rho(f^{d-1})=1$$. A faithful, yet non-regular rep is $$\rho(j)=e^{2 \pi i j / d}$$ where $$j$$ enumerates group elements e.g., $$\rho(e)=1$$, $$\rho(f)=e^{2 \pi i / d}$$, ..., $$\rho(f^{d-1})=e^{2 \pi i (d-1) / d}$$. A regular representation (which is always faithful) has size $$d$$ and corresponds to $$d \times d$$ permutation matrices, which I display below for $$d=4$$ 

$$
\begin{array}{cccc}
\rho(e)=\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
&
\rho(f)=\begin{pmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0
\end{pmatrix}
&
\rho(f^2)=\begin{pmatrix}
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{pmatrix}
&
\rho(f^3)=\begin{pmatrix}
0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{pmatrix}
\end{array}
$$

<blockquote class="note">
  <b>Upshot:</b> A group of symmetries can be represented as matrices with different dimensions. 
</blockquote>


### Some intutition: why can symmetries be helpful for neural nets?

Okay, enough about representations. Before we dig into a more details of marrying symmetries and neural nets, let's take a step back and play devil's advocate: why should symmetries be helpful for neural net predictions in the first place? One extra intuitive perspective I want to offer is the following: presence of symmetries in a problem suggests a further restriction on the **hypothesis space** that a generic non-symmetric neural net can explore. In other words, there is no need to explore non-symmetric parts of the loss landscape given that we know that the solution does not lie there! By explicitly imposing symmetries on neural nets (as in the weight sharing approach discussed in one of the [later sections](#weight-sharing)) we reduce the size of the hypothesis space thus improving generalization [[Goodfellow+ 2016]](#references).

### Equivariance and invariance
Cool so now we have a slightly better understanding of what we mean by symmetries and why they may be helpful for neural nets. Physicists are super used to thinking about symmetry actions as leaving the object **invariant**. In ML people talk quite a bit about “a map being **equivariant** with respect to representations of a group of symmetries”. What does it mean?

Take an image of a cat. We want neural network output to be *invariant* under translating a cat by moderate amounts (e.g., ones not taking it outside of an image). Mathematically it means we would want $$f(gx)=f(x)$$ for all symmetry group elements $$g \in G$$ from translational symmetry groups for an overall neural network represented as a function $$f: x \mapsto f(x)$$. Let’s look inside the neural network though: in each layer $$k \in \{1,\dots,K\}$$ it will extract some features, for simplicity let's say, eyes. As we shift a cat on an image we would like the eye features to shift as well rather than stay put - in other words, we want such transformation to be "equally-varying with a cat" i.e. to be “equivariant” rather than “invariant”. It can be more mathematically written as $$f_k(gx)=gf_k(x)$$.  Therefore, an **equivariant neural network** is typically built by stacking up layers, each equivariant under actions of a certain group: $$f_k(gx)=gf_k(x)$$ for a neural net layer $$f_k$$. Such series of equivariant layers would be typically finished by a final, **invariant** layer $$f_K(gx)=f_K(x)$$ at the very end if the output of the neural net has properties of a scalar (e.g., it is a label). 

Let’s try to be slightly more formal to unconceal more. Let’s think of $$k-$$th layer of the neural network as mapping between two vector spaces $$V_{k}$$ and $$V_{k+1}$$ i.e. $$f_k: V_{k} \rightarrow V_{k+1}$$. Consider also symmetry group with elements $$g \in G$$ and its representation on $$V_k$$: $$\rho_i(g)$$ and on $$V_{k+1}$$ i.e. $$\rho_o(g)$$ (two representations not need to be the same, particularly if spaces $$V_{k}$$ and $$V_{k+1}$$ have different dimensionality). Now consider you first act on $$V_k$$ with a symmetry operation $$\rho_i(g)$$ e.g., take an image of a cat $$x$$ (an element of vector space $$V_k$$) and shift it by some amount (action of $$\rho_i(g)$$ on $$x$$). For equivariance we require $$f_k(\rho_i(g) x)= \rho_o(g) f_k(x) \ \forall x \in V_k \ \forall g \in G$$ i.e. the output transforms under the same group of symmetries as the input (perhaps with a different representation though). This can be depicted as the diagram in Fig. 4 (left panel). Note that in this sense invariance is a special case of equivariance where we set $$\rho_o(g)= \mathbb{1} \ \forall g \in G$$ i.e. outer representation is trivial: then $$f_k(\rho_i(g) x)= f_k(x) \ \forall x \in V_k \ \forall g \in G$$ : see Fig. 4 (right panel). 

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_symm_neural_net_equivariance.png" width="700" loading="lazy"/></p>
Fig. 4: Handwavy way of writing so called "commutative diagrams" of equivariant and invariant transformations.  
{:.figcaption}

<blockquote class="note">
  <b>Upshot:</b> Symmetric neural networks are typically constructed by stacking up equivariant neural networks layers. 
</blockquote>


### A word of caution: CNNs are not fully translationally invariant?

An intermission to make a full circle: to complicate your life even further, at this stage, I need to warn you that typical CNN is **not** fully invariant under translation symmetry (in contrast to what many people are claiming!). This is due to a combination of two things: (i) subtle effects related to aliasing and (ii) wide-spread presence of final dense layer at the very end of the CNN. We can already understand the latter: dense layer at the end means that shifting the features within that layer will *change* the output of the layer in general thus violating layer-wise equivariance requirement[^6]. The former we will try to understand a bit more in the [section on limitations of symmetric networks](#references). For now, to spice things up even more, it [turns out](https://arxiv.org/pdf/2003.07064) CNNs can often learn absolute spatial locations for objects (despite being translationally invariant). Read through the paper if you are interested![^7]


## How to explicitly teach symmetries to a neural net?

Good, now we know what do we mean by symmetries and recognize notions of symmetric maps corresponding to equivariance and invariance. We argued that we should stick to equivariant layers of the network for extracting features often followed by a final invariant layer. Now, how can one ensure that the neural network layer is equivariant? Broadly speaking, there are at least two different ways. 

### Data augmentation
In a data-driven setup, the simplest and often the cheapest way of ensuring that outputs of the neural network are symmetric under transformation is **data augmentation**. There, if data is symmetric under group $$G$$, one just transforms each example in the dataset by all possible transformations in $$G$$ and artificially boosts the size of the dataset in this way. This is all hoping that for unseen examples neural network will learn that as one transforms the input with a group operation, the output should change accordingly (see Fig. 5 below for another cat-cartoon). 

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_symm_neural_net_data_augmentation.png" width="700" loading="lazy"/></p>
Fig. 5: A cartoon for rotation symmetry data augumentation.  
{:.figcaption}

This strategy is surprisingly successful having been recently applied e.g., within AlphaFold 3 [[Abramson 2024+]](#references). Symmetry-augmenting data only increases the training (but not evaluation time) with respect to baseline architecture by a factor of $$\mathcal{O}(\vert G \vert)$$ (assuming we are talking discrete group here). Data augmentation also has some theoretical underpinning in terms of reducing generalization error: see [[Wang+ 2022b]](#references) and [[Chen+ 2019]](#references). I should warn you, however, that this approach is data-centric and can't be straightforwardly applied to non-data-driven problems. For instance, data augmentation does not directly apply whenever we use neural networks as a powerful optimization ansatze as in [neural quantum states](https://dom-kufel.github.io/blog/2024-06-17-neural_quantum-intro/) - since there is simply no data there! Therefore, when in the next blogpost we will turn to quantum many-body physics, we will have to use some alternative ways of teaching neural net symmetries, with one of them presented below.

<blockquote class="note">
  <b>Upshot:</b> Neural networks can be taught to be symmetric through data augmentation. 
</blockquote>

### Weight sharing

In another approach, known as **weight sharing**, instead of boosting the dataset one restricts the neural net architecture in a certain way. My favorite construction, which is quite general is that proposed by [[Finzi+ 2021]](#references) and coined equivariant multi-layered perceptron (equivariant MLP). It works for both discrete and continuous (Lie) groups and reduces to other commonly used group-equivariant frameworks such as G-convolutional [[Cohen&Welling 2016]](#references), G-steerable [[Cohen&Welling 2016b]](#references) or deep set [[Zaheer+ 2017]](#references) architectures. I should mention that these approaches are already *generalizations* of CNNs to more general groups beyond translation! 

The main idea of equivariant MLPs is very simple: if you want linear layer of a neural network $$f: V \rightarrow W$$ to be equivariant with respect to input and output representations $$\rho_i$$ and $$\rho_o$$ of the group $$G$$, given a general form of the weight matrix $$W$$, just solve the following set of equations for $$W$$: $$\rho_o (g) (W x) = W (\rho_i (g) x) \ \forall g \in G \ \forall x \in V$$. Let's see it on a super simple example, which however demonstrates essential steps for solving many more difficult problems!

#### 1D translation equivariance: linear layers

We will consider a <a href="#groups-and-representations">regular representation</a> of translational symmetry group of 4-pixel-wide 1D images. Suppose pixels can only be black or white ($$0$$ or $$1$$) and we try predicting e.g., total parity of the input with a simple neural net (although it does not matter precisely what the task is!). The translation group presentation is $$G=\langle T \vert T^4 = \mathbb{1} \rangle$$ i.e. the group has a single generator $$T$$ fulfilling relation $$T^4=\mathbb{1}$$. Thus for an $$N=4$$ pixel image we have $$\vert G \vert = N$$ and we will choose $$\rho_i = \rho_o \ \forall g \in G$$ with the following representation of translation to the right by one element written as a $$4 \times 4$$ matrix

$$
T = \begin{pmatrix}
0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{pmatrix}
$$

**Exercise:** By acting on a trial column vector e.g., $$ x = \begin{pmatrix} x_1 & x_2 & x_3 & x_4 \\ \end{pmatrix}^T $$ convince yourself that matrix $$T$$ indeed shifts $$x$$ entries by $$1$$ to the right.  
{:.message}

Good, now consider any vector $$x$$ in an input space $$V$$. Following the prescription above we should write $$W T x = T W x \ \forall x \in V$$ and similarily for all (non-identity) powers of $$T$$ (corresponding to translations by more than 1 pixel). Since the expression needs to hold for all $$x \in V$$ this means we can write simply $$WT = TW$$ etc. which yields

$$
\begin{align}
[W,T] &= 0, \\
[W,T^2] &= 0 \\
[W,T^3] &= 0.
\end{align}
$$

i.e. we require commutation of a weight matrix $$W$$ with all of the group elements. It is straightforward to see that only one of these equations is independent and thus it is enough to demand $$[W,T] = 0$$. Is this to be expected? Yes, as we will see below from the complexity bounds of the method, this simplification is more general: computational cost of the method does not scale with $$\vert G \vert$$ but instead with the number of generators; this is similar to G-steerable networks [[Cohen&Welling 2016b]](#references) but in contrast to e.g., plain G-convolutional networks of [[Cohen&Welling 2016]](#references). Now going back to $$[W,T] = 0$$: for our case it can be easily solved for a general form of $$W$$ under such constraints thus introducing weight sharing (see Fig. 6 below)

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_symm_neural_net_weight_sharing.png" width="600" loading="lazy"/></p>
Fig. 6: Equivariance of a linear transformation can be thought within the framework of weight sharing. 
{:.figcaption}

To summarize: the interpretation is very simple: we want a linear transformation to be symmetric under certain group of symmetries and we appropriately restrict the weights of the matrix to belong to a few classes (here $$\textrm{# of classes}=\vert G \vert$$) within which the weights are shared, thus automatically fulfilling equivariance constraints! 

**Exercise:** Check the above calculation yourself and verify that such constructed $$W_{shared}$$ indeed fulfills equivariance constraints.  
{:.message}

#### 1D translation equivariance - how do they relate to convolutional nets?

I have motivated the use of symmetries partly based on the success of convolutional neural nets (CNNs) imposing translational symmetry. This can be thought as a special case of the weight sharing approach described above. A little reminder on convolutional neural nets: therein we use **convolutional kernels** (also known as filters) of size $$k$$ acting on the 1D input data "image" $$x$$ (a more usual case of a 2D CNN can be worked out similarily). To impose locality one would choose $$k \sim \mathcal{O}(1)$$ and if we want it to span entire image globally then we would choose $$k = N$$ (for a 1D "image" of size $$N$$). Since we want to delineate effects of locality and translational symmetry, we will assume the latter: each kernel will be of size $$k = N$$ thus not imposing any notion of locality. Now CNNs work by shifting a kernel through an image (say to the right) with increments (known as a stride $$s$$) which we will assume to be $$1$$ (to keep the dimensionality of the transformation with a goal of matching the weight sharing calculation from the previous section). This can be described by the following equation:

$$
y_i = \sum_{j \in \mathbb{Z}_N} w_{i} x_{i-j}  
$$
where $$w = \begin{pmatrix} w_1 & w_2 & w_3 & \dots & w_N \\ \end{pmatrix}^T$$ is a kernel vector (of size $$N$$ as described above). 

After renaming the summation variable we get:
$$
y_i = \sum_{j \in \mathbb{Z}_N} w_{i+j} x_{i}  = [W x]_i
$$
where in the final equation we stack up vectors $$w$$ in the rows of the matrix $$W$$ such that in row $$i$$ they are shifted by $$i$$ to the right (for $$N=4$$):

$$
W = \begin{pmatrix}
w_1 & w_2 & w_3 & w_4 \\
w_4 & w_1 & w_2 & w_3 \\
w_3 & w_4 & w_1 & w_2 \\
w_2 & w_3 & w_4 & w_1
\end{pmatrix}
$$

Does it look familiar? **Yes!** It is the same matrix we have obtained using weight sharing approach. It is straightforward to generalize this approach to an arbitrary value of $$N$$ and for local kernels[^8] $$k \sim \mathcal{O}(1)$$. The latter generalization can be found in Fig. 7. This formally establishes equivalence between convolutions in 1D and weight sharing in an equivariant MLP approach of [[Finzi+ 2021]](#references). This is a special case of a much more general statement regarding reduction of the weight sharing approach to well-studied group-convolutional neural networks of [[Cohen&Welling 2016]](#references). Nice!

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_symm_neural_net_local_kernel_cnns.png" width="800" loading="lazy"/></p>
Fig. 7: Equivalence of 1D CNNs with local kernels $$k=3$$ and weight sharing matrix after imposing locality on the latter.
{:.figcaption}

<blockquote class="note">
  <b>Upshot:</b> For translational symmetry, weight sharing approach reduces to convolutional neural nets. 
</blockquote>

<!-- - locality imposing
- stride imposing? -->

#### Non-linearities, biases and all that

Okay, so imposing equivariance is all about weight sharing. But you might argue: neural nets are affine transformations interspersed by non-linearities, and so far we have shown how to make only linear layers to be equivariant. Well, linear to affine generalization is easy: one can think of an affine transformation as a linear transformation in a higher dimensional space via an ["augmented matrix"](https://en.wikipedia.org/wiki/Affine_transformation). 

How about non-linearities? In other words is it automatically true that $$\sigma(\rho_i (g) x) = \rho_o (g) \sigma (x) \ \forall x \in V \ \forall g \in G$$? Here the situation is much more subtle: for point-wise non-linearities (i.e. acting on each neuron separately e.g., $$SELU$$, $$tanh$$ etc.) and regular representations, any choice of such point-wise non-linearity preserves equivariance. Intuitively this is because regular representations of any group will only lead to a permutation of the neurons in a layer and therefore acting on neurons **point-wise** yields the same value as first acting with non-linearities and permuting afterwards. 

However, a choice of the non-linearity is much more restricted when representations are non-regular. This is because point-wise non-linearities will not be automatically equivariant under non-regular representations which do not simply act as permutations on the input. We need to ensure that $$\sigma(\rho_i (g) x) = \rho_o (g) \sigma (x) \ \forall x \in V \ \forall g \in G$$ which is not automatically fulfilled by typical point-wise non-linearities. The resolution, assuming unitary (or orthogonal) representations is to use e.g., norm non-linearities i.e. $$\sigma(x)= x \sigma( \vert x \vert^2)$$ where $$x$$ now is treated like a vector representing values on all neurons within the layer and does NOT act pointwise. Fulfillment of the equivariance condition comes from invariance (due to norm-preservation) of $$\sigma( \vert \rho_i x \rho_i^T \vert^2)=\sigma( \vert x \vert ^2)$$ and equivariance of the $$x$$ term in front of it. Furthermore, other choices of non-linearities are possible, e.g., [[Weiler+ 2018]](#references) proposed using gated non-linearities or tensor product non-linearities.  

**Exercise:** In case it is not obvious, show that unitary matrices do not change the norm. 
{:.message}

#### Limitations of equivariant MLPs

Wrapping up a section on equivariant MLPs, what are their limitations as a way of imposing equivariance on a neural net? In practice, this approach is only limited by a relatively high $$\mathcal{O}((M+D)m^3)$$ complexity of numerically solving for allowable weight matrix $$W$$ where $$m$$ is a size of an input (e.g., number of pixels in the image), $$M$$ ($$D$$) is a number of *generators* of discrete (continuous) symmetries. In cases of very large images (and for symmetry groups which can be decomposed into product of translations and something else) G-steerable convolutions [[Cohen&Welling 2016b]](#references) are probably a better approach. What remains curious is also an extra degree of freedom in constructing equivariant MLPs: a choice of representations of symmetry groups in hidden layers. Although, input and output layers have their representations fixed, one can tune representations of hidden layers of the networks freely: are there any better or worse choices? Does e.g., a degree of faithfulness of the representation matter? These questions will perhaps be addressed by further research. 

<blockquote class="note">
  <b>Upshot:</b> Symmetric neural networks can be constructed through weight sharing in linear layers and equivariant non-linearities. 
</blockquote>

## Limitations of symmetric neural nets and possible remedies

We have talked quite a bit about different flavors of teaching neural nets symmetries assuming a certain pristine setup, for instance: (i) presence of perfect symmetries in the data and (ii) not being careful about properly treating finite image resolutions in our considerations - see Fig. 8! Let's try to relax these assumptions now and see if equivariance (even beyond equivariant MLPs studied above) can still be helpful! 

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_symm_neural_net_approx_symmetries_aliasing.png" width="800" loading="lazy"/></p>
Fig. 8: Two limitations of symmetric neural nets: non-fully symmetric data (left panel) and aliasing phenomenon (right panel).
{:.figcaption}

### Data symmetries are not perfect

World isn't perfect right? Data isn't neither. One way to address it is to relax requiring strict equivariance of models to only **approximate equivariance**. This amounts to requiring $$f(g x) \approx g f(x)$$ instead. The right amount of symmetry violation can be figured out by the neural network itself. The simplest way of achieving it is simply through combining non-symmetric and symmetric layers within a neural network architecture (e.g., fully equivariant layers followed by non-symmetric layers). There exist several more complicated constructions proposed in [[Finzi+ 2021b]](#references) and [[Wang+ 2022]](#references) which show advantages over simpler constructions on some datasets. In fact, we will see that requiring neural networks to be approximately symmetric instead of fully symmetric can be also highly beneficial for solving some quantum many-body physics problems. More on this in the next blogpost! 

### Aliasing and equivariance
Before we conclude one last interesting fact: surprisingly, it turns out that in many cases vision transformers without any explicit symmetry encodings can be more equivariant than the CNNs which have the symmetries baked in! How come can it be the case? The culprit was already briefly mentioned in [one of the earlier sections](#a-word-of-caution-cnns-are-not-fully-translationally-invariant): it is the effect known as **aliasing**. 

I should start by mentioning the obvious: our world, when projected to a plane, has continuous $$\mathbb{R} \times \mathbb{R}$$ symmetry and not $$\mathbb{Z}_L \times \mathbb{Z}_L$$ symmetric as assumed in CNNs applied to an $$L \times L$$ image. In other words, a finite $$L$$ means a finite image resolution and therefore instead of a full $$\mathbb{R} \times \mathbb{R}$$ symmetry (where translation by any vector is allowed), we restrict ourselves to multiples of translations by 1 pixel only but not smaller[^9]. More importantly, but in the similar spirit, when we downsample the size of the image (e.g., when using a stride $$s=2$$ in a CNN) we break the $$\mathbb{Z}_L \times \mathbb{Z}_L$$ group to a smaller one (e.g., $$\mathbb{Z}_{L/2} \times \mathbb{Z}_{L/2}$$). 

What significance does it have? Well, it implies that only shifts which are multiples of the downsampling factor will keep CNN output equivariant [[Azulay&Weiss 2019]](#references). In this sense, in CNNs we are imposing often an *incomplete* group of symmetries. In fact, it was observed, to the surprise of many, that in multiple cases classification accuracy for a given image in a dataset is highly sensitive to shifts by certain vectors (not being a multiple of the downsampling factor i.e. shifts outside of our imposed group), yet insensitive to the others (multiples of the downsampling factor) [[Zhang 2019]](#references). Why does it happen? It can be explained through a concept of aliasing. 

So what is aliasing? It happens whenever we *undersample* certain high frequency signal which appears then to us as a lower frequency signal (see Fig. 5, right panel). 
Frequencies come into the picture[^10] by thinking of a Fourier transform of an image: instead of representing information contained within an image in the real space we will think about it in a frequency space[^11]. Finite resolution of an image introduces minimum $$-L/2$$ and maximum $$L/2$$ frequency in a discrete Fourier transform of an image. One can therefore think of images as if they were sampling continuous space with freqencies limited by $$f_s=L/2$$. From [Nyquist-Shannon sampling theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem) it follows that information in the true continuous signal is retained fully only if the image contained signals only below $$f_s/2$$ (known as **Nyquist frequency**) and thus any information above that frequency may be lost through aliasing. 

Summarizing, quite expectedly we lose information about continuous space by dealing with discrete images. What is, however, more surprising is that by the same argument we also lose information further during image processing with a CNN, all through aliasing phenonomenon: 
1. Within downsampling layers. This is because when we downsample $$ L \rightarrow L/2$$, by the sampling theorem, we are effectively losing frequency signal in range $$(L/4,L/2)$$. 
2. Within point-wise non-linearities. Why? Because, as shown by [[Karras+ 2021]](#references), point-wise non-linearities (especially less smooth ones) often introduce more high frequency components to the image processing which can push the information beyond the **Nyquist frequency**[^12].

Quite naturally, aliasing also breaks equivariance as explicitly shown by [[Gruver+ 2022]](#references). Without aliasing, shifting by a vector $$(v_x,v_y)$$ in a real space corresponds to an extra phase in the Fourier transform[^13] 

$$G(f_x,f_y) \mapsto G(f_x,f_y) e^{-2\pi i(v_x f_x + v_y f_y)}$$

However, when signal is aliased then one instead applies 

$$G(f_x,f_y) \mapsto G(f_x,f_y) e^{-2\pi i(v_x \textrm{Alias} (f_x) + v_y \textrm{Alias}(f_y))}$$ 

which will apply incorrect shifts for frequencies $$f_x,f_y > f_s /2$$ i.e. where $$\textrm{Alias}$$ function acts non-trivially. One can directly link these incorrect shifts to introducing equivariance error as beautifully shown in a Theorem 1 of [[Gruver+ 2022]](#references). This also explains the earlier results [[Zhang 2019]](#references) on CNNs respecting only shifts by multiples of downsampling factors. 

Final remark: I should mention that this picture holds way beyond translational symmetries (e.g., similar behavior was observed to rotations, scalings etc.), thus making aliasing to be quite a general phenomenon! People have figured out[^14] some architectural ways of mitigating aliasing effects by applying anti-aliasing filters [[Zhang 2019]](#references), although, in practice, what matters the most for the improved equivariance seems to be an increased model scale and dataset size [[Gruver+ 2022]](#references). In practice,the above-mentioned facts allow non-inherently symmetric architectures such as vision transformers to be more equivariant than CNNs (especially with extra data augmentations). However, as of early 2025, it seems that there are still many places where intrinsically symmetric networks can have a significant edge over transformers - and in the next blogpost I'd like to argue that this corresponds to multiple problems in quantum physics! 

<!-- Since an image, call it $$G(x,y)$$ (with $$0\leq x,y \leq1$$) has size $$L \times L$$ it implies that it has a *discrete* Fourier transform:

$$
G_{x,y} = \sum_{n,m} G_{nm} e^{i(x n + y m)}
$$
and $$n,m \in \{-L/2,-L/2+1,\dots,L/2\}$$ i.e. maximum frequencies $$n,m$$ are limited by the size of the image. Intuitively, this is because large frequencies encode very small spatial features, which cannot be captured when space discretization is too coarse. -->


<!-- <details>
<summary><b>ML BOX 1:</b> Notation, quantum states and inner products. </summary>
<div markdown="1">
<span style="font-size:0.85em;"> In quantum mechanics, the state of a system, such as a collection of "qubits" (spins $$1/2$$) is represented as a vector $$|\psi \rangle$$ in a tensor product Hilbert space. For a system with $$N$$ qubits, this Hilbert space has a dimension $$2^N$$. Any vector in this space might be decomposed in a complete basis of length $$N$$ combinations of 0’s and 1’s.  For instance $$|000\rangle = |0\rangle \otimes |0\rangle \otimes |0\rangle$$ for $$N=3$$ corresponds to a basis vector $$e_0=(1,0,0,0,0,0,0,0)$$. To denote vectors in a Hilbert space $$\mathcal{H}$$ physicists often use bra-ket notation where "ket" is denoted by $$|\psi \rangle \in \mathcal{H}$$ and dual vector "bra" by $$\langle \phi | \in \mathcal{H}^{*}$$. In such notation, an inner product becomes $$\langle \phi | \psi \rangle \in \mathcal{C}$$. A quantum mechanical expectation value of an operator $$Q$$ in state $$|\psi \rangle$$ then can be written as $$\langle \psi | Q \psi \rangle$$. Throughout we assume working with an orthonormal basis $$\langle i|j\rangle = \delta_{ij}$$ thus e.g., $$\langle 000 | 001 \rangle = \langle 0 | 0 \rangle \langle 0 | 0 \rangle \langle 0 | 1 \rangle = 0$$. If you feel uncomfortable with using bra-ket notation just think of $$|\psi \rangle$$ as a $$2^N$$ dimensional complex vector $$\psi$$ (decomposable into an orthonormal complete basis as $$\psi = \sum_i \psi_i e_i$$ where $$e_i^{\dagger} e_j = \delta_{ij}$$), inner product $$\langle \phi | \psi \rangle$$ as $$\phi^{\dagger} \psi$$ where $${\dagger}$$ denotes conjugate transpose, and $$\langle \psi | Q \psi \rangle$$ as a quadratic form $$\psi^{\dagger} Q \psi$$. </span>
</div>
</details> -->
<!-- <br> -->


## Outlook

I hope you have enjoyed reading about symmetric neural networks! We have talked about how symmetries can be described as mathematical groups and represented by matrices. We have distinguished two senses in which we mean symmetric: equiviariance, appropriate for vector-like features (cat's whiskers features should translate as the cat translates), and invariance, applicable for scalars (e.g., output "cat" label is a "cat" label after translation of the input). Then we have discussed two ways of ensuring neural nets preserve desired symmetries: data augmentation and weight sharing. Finally, we discussed the limitations of these approaches: for weight sharing a finite resolution of images and for data augmentation an increased training cost and limited applicability. Phew, we have covered quite a bit of stuff — high-five for making it here!

In the next blogpost we will build up on this knowledge and see how to connect symmetric neural nets research in ML to the one in quantum physics, highlighting some surprising parallels. Stay tuned!

## FAQ
1. **How do I learn more about equivariance, along with a more rigorous mathematical treatment?** For a good, simple, overview I would recommend reading [this](https://arxiv.org/abs/2205.07362). For a more in-depth and rigorous (yet still pedagogical) treatment I invite you to read through an excellent [Maurice Weiler book](https://maurice-weiler.gitlab.io/cnn_book/EquivariantAndCoordinateIndependentCNNs.pdf). 
2. **The fact that CNNs are not so equivariant still puzzles me! Where can I read more?** To read further about limitations of symmetric neural nets in context of CNNs and aliasing I highly recommend [Marc Finzi's PhD thesis](https://cs.nyu.edu/media/publications/Marc_Finzi_Thesis__12_.pdf) as well as following papers: [[Zhang 2019]](#references), [[Karras+ 2021]](#references), [[Azulay&Weiss 2019]](#references) and [[Gruver+ 2022]](#references) as well as [[Biscione & Bowers 2021]](#references), [[Mouton+ 2021]](#references),; regarding approximately symmetric networks I recommend [[Wang+ 2022]](#references).
3. **Which methods can I most feasibly apply to quantum many-body physics context?** Excellent question, this is precisely the content of the second blogpost coming soon! 

**Acknowledgements**: I thank Shubhendu Trivedi, Jack Kemp, Marc Finzi, Maurice Weiler, Rui Wang, DinhDuy Vu, and Max Welling for many fun discussions on symmetric neural nets. Feel free to contact me at dkufel (at) g.harvard.edu if you have any questions or comments! 
{:.message}

## References

[Abramson+ (2024)](https://www.nature.com/articles/s41586-024-07487-w_reference.pdf) Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A.J., Bambrick, J. and Bodenstein, S.W., 2024. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, pp.1-3.

[Azulay&Weiss (2019)](https://arxiv.org/pdf/1805.12177) Azulay, A. and Weiss, Y., 2019. Why do deep convolutional networks generalize so poorly to small image transformations? Journal of Machine Learning Research, 20(184), pp.1–25.

[Biscione & Bowers 2021](https://www.jmlr.org/papers/volume22/21-0019/21-0019.pdf) Biscione, V. and Bowers, J.S., 2021. Convolutional neural networks are not invariant to translation, but they can learn to be. Journal of Machine Learning Research, 22(229), pp.1-28.

[Cohen&Welling (2016a)](https://arxiv.org/pdf/1602.07576) Cohen, T.S. and Welling, M., 2016. Group Equivariant Convolutional Networks. Proceedings of The 33rd International Conference on Machine Learning, 48, pp.2990–2999.

[Cohen&Welling (2016b)](https://arxiv.org/pdf/1612.08498) Cohen, T.S. and Welling, M., 2016. Steerable CNNs. arXiv preprint arXiv:1612.08498.

[Chen+ (2020)](https://arxiv.org/pdf/1907.10905) Chen, S., Dobriban, E. and Lee, J.H., 2020. A group-theoretic framework for data augmentation. Journal of Machine Learning Research, 21(245), pp.1-71.

[Finzi+ (2021)](https://arxiv.org/pdf/2104.09459) Finzi, M., Welling, M., and Wilson, A.G., 2021. A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups. Proceedings of the 38th International Conference on Machine Learning, 139, pp.3318–3328.

[Finzi+ (2021b)](https://arxiv.org/pdf/2112.01388) Finzi, M., Benton, G. and Wilson, A.G., 2021. Residual pathway priors for soft equivariance constraints. Advances in Neural Information Processing Systems, 34, pp.30037-30049.

[Goodfellow+ (2016)](https://www.deeplearningbook.org/) Goodfellow, I., Bengio, Y., and Courville, A., 2016. Deep Learning. MIT Press.

[Gruver+ (2022)](https://arxiv.org/pdf/2210.02984) Gruver, N., Finzi, M., Goldblum, M. and Wilson, A.G., 2022. The lie derivative for measuring learned equivariance. arXiv preprint arXiv:2210.02984.

[Karras+ (2019)](https://proceedings.neurips.cc/paper/2021/file/076ccd93ad68be51f23707988e934906-Paper.pdf) Karras, T., Aittala, M., Laine, S., Härkönen, E., Hellsten, J., Lehtinen, J. and Aila, T., 2021. Alias-free generative adversarial networks. Advances in neural information processing systems, 34, pp.852-863.

[LeCun+ (1995)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e26cc4a1c717653f323715d751c8dea7461aa105) LeCun, Y. and Bengio, Y., 1995. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 3361(10), p.1995.

[Mouton+ (2021)](https://arxiv.org/pdf/2103.10097) Mouton, C., Myburgh, J.C. and Davel, M.H., 2020, December. Stride and translation invariance in CNNs. In Southern African Conference for Artificial Intelligence Research (pp. 267-281). Cham: Springer International Publishing.

[Wang+ (2022)](https://arxiv.org/pdf/2201.11969) Wang, R., Walters, R. and Yu, R., 2022, June. Approximately equivariant networks for imperfectly symmetric dynamics. In International Conference on Machine Learning (pp. 23078-23091). PMLR.

[Wang+ (2022b)](https://arxiv.org/pdf/2206.09450) Wang, R., Walters, R. and Yu, R., 2022. Data augmentation vs. equivariant networks: A theory of generalization on dynamics forecasting. arXiv preprint arXiv:2206.09450.

[Weiler+ (2019)](https://arxiv.org/pdf/1911.08251) Weiler, M. and Cesa, G., 2019. General E(2)-Equivariant Steerable CNNs. Advances in Neural Information Processing Systems, 32.

[Zaheer+ (2017)](https://arxiv.org/pdf/1703.06114) Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R.R., and Smola, A.J., 2017. Deep Sets. Advances in Neural Information Processing Systems, 30.

[Zhang (2019)](http://proceedings.mlr.press/v97/zhang19a/zhang19a.pdf) Zhang, R., 2019, May. Making convolutional networks shift-invariant again. In International conference on machine learning (pp. 7324-7334). PMLR.