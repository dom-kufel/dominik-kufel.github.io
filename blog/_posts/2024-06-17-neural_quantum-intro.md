---
layout: post
title: Neural networks for quantum many-body physics
description: >
  A quick intro to the neural networks for quantum many-body physics. Basic quantum background encouraged but not necessary.
sitemap: false
---

<!-- 2023-01-19 -->

<!-- related_posts:
  - /blog/_posts/2023-07-04-toric_code-lre.md -->

<!-- image: /assets/img/blog/summary.png -->


*[staying]: Of course assuming you have NOT directly scrolled here, haha!
 
[^1]: See e.g., [this](https://www.nature.com/articles/s41598-023-45837-2) for evaluating the GPT-4 and GPT-3.5 for opthalmology self-assesment program. 
[^2]: This is one of the key differences between neural networks for quantum setup and typical supervised learning: we do not need to worry about generalization, only about minimizing the loss function as much as possible. 
[^3]: You might ask why it is valid to make this assumption? Well, in general it is not and for some problems this assumption is violated and introduces a bias to the sampling method! We will further discuss it later.    
[^4]: In case you are unfamiliar with Metropolis-Hastings algorithm: the key idea is to (i) start from a random bit string $$s_0$$ (ii) generate another bit string $$s_1$$ by applying an update rule to $$s_0$$ (e.g., flip a bit at a random location), (iii) calculate acceptance probability for $$s_1$$ (for simplest case of symmetric update rule as in the example above as: ) and (iv) draw a number uniformly from range $$[0,1]$$, if it is below or equal to the acceptance probability from (iii) then accept, if not then reject the $$s_1$$ configuration and draw a new bit string (v) repeat to construct a Monte Carlo Markov chain. 
<!-- $$\min(1,\frac{|\psi_{s_1}|^{2}}{|\psi_{s_0}|^{2}})$$ -->
[^5]: Ergodic in our case means that such update rule allows us to explore all bit string space.
[^6]: To fully rule out sampling inefficiency for stoquastic Hamiltonians I guess one would have to further formally show that sampling is efficient along a particular optimization pathway as well. By this I mean the following: during neural quantum state optimization one will start from a certain set of parameters $$\theta_0$$ and try converging them to the ground state with $$\theta_*$$ (provided neural network is expressive enough). Although efficient sampling of ground state with $$\theta_*$$ is then guaranteed, sampling of all intermediate states between $$\theta_0$$ and $$\theta_*$$ is not. It is not immediately clear to me however if efficient sampling everywhere along the optimization pathway is even important! 
[^7]: Such increased computational cost is prohibitive for models with a huge number of parameters and thus classical counterpart of a natural gradient is rather rarely used in practice in classical machine learning!
[^8]: In practice, step 2 and step 3 would be done together when e.g., using automatic differentiation. 
[^9]: Note that this is not true for generic variational ansatze! 
[^10]: Just a reminder: evaluating observables on tensor networks such as MPS or PEPS requires contracting a tensor network. Although efficiently contracting MPS is not a problem, contracting PEPS is in general a #P-hard problem (think exponentially hard). The efficient contractibility requirement is helpful for the proof in the linked reference (efficiently contractible tensor networks are mapped to neural networks that perform efficient contractions). 
[^11]: You may think of restricted Boltzmann machines as a feed-forward 1 hidden layer neural network architecture with $$\log \cosh$$ non-linearites and product aggregation in the final layer.  
[^12]: Another, more direct, intuition comes from studies of convolutional neural networks (or more specifically so called "convolutional arithmetic circuit) by <a href="#references">*[Sharir+ (2019b)]*</a>. These operate based on sliding kernels of size $$K \times K$$ (in 2D) each time moving by stride of size $$S$$, applying non-linearities and repeating this operation in $$L$$ layers. It turns out that for 2D systems the maximum amount of entanglement of a subsystem of size $$\alpha \times \alpha$$ as represented by the above neural network with stride $$S=1$$ is lower bounded by $$L K \alpha$$. This implies that as long as $$L K > \alpha$$, such neural network architecture can support volume law entanglement (only with $$L K^2$$) parameters! Authors point out that the key component which allows for efficient representation of volume law comes from information re-using when subsequent kernel moves overlap (e.g., tree tensor network would correspond to $$S=K$$ which corresponds to non-overlapping kernels). Fun fact: handwavingly this condition might be turned upside down and applied to **classical ML** in an interresting recent paper <a href="#references">*[Alexander+ (2023)]*</a>: successfull application of using locally connected neural networks (such as convolutional neural networks with $$\mathcal{O}(1)$$ size kernels) requires quantum entanglement of the underlying data to be "small"!
[^13]: By this I mean few things: first existence of efficient libraries benefiting from automatic differentatiton, vectorization and just-in-time compilation (such as JAX) and second existence of hardware tailored for collecting multiple passes through a neural network in parallel such as GPUs or TPUs. 
[^14]: In 1D application of neural networks does not make so much of sense since matrix product state methods just work incredibly well! 
[^15]: DMRG algorithm has an unfavorable $$\mathcal(N \chi^3)$$ [runtime complexity](https://link.springer.com/article/10.1140/epjb/s10051-023-00575-2) (where $$\chi$$ is a bond dimension). 
[^16]: This seems like more of a general pattern: [recent results](https://arxiv.org/abs/2404.19023) suggest that computational difficulty of tensor network contraction to an extent also suffers from a sign-problem. 
[^17]: Note that presence of complex phases in the outputs of the neural networks is another difference between neural quantum state and more traditional machine learning setup. Although there are some works e.g., <a href="#references">*[Arjovsky+ (2015)]*</a> in such setup and operate on complex parameters, they point out that the performance of such models is sensitively depends on the choice of non-linearities for complex numbers. Alternatively one can work with real parameters and create two networks: one learning a real phase $$\phi$$ and the other a real amplitude $$A$$ (only later combining them by $$A e^{i \phi}$$), but this approach has only a limited success <a href="#references">*[Szabo&Castelnovo (2020)]*</a>
[^18]: Approximate symmetries naturally appear in context of Hamitlonians perturbed away from the symmetric fixed points (whcih corresponds to a typical experimental setup!), and includes e.g., a class of quantum spin liquids. See [our paper](https://arxiv.org/pdf/2405.17541) for more details!
[^19]: Although, presence of regularization also decreases the accuracy of the neural quantum states for ground state search as well, it is much less of an issue there. The reason is that $$\epsilon$$ regularization parameter(see <a href="#energy-optimization">optimization section</a> for definition), interpolates between quantum natural gradient (aka imaginary time evolution) and "standard" gradient update. Therefore even in $$\epsilon \rightarrow \infty$$ limit, such optimization is admissable for ground states (simply ignoring curvature information). This, of course, contrasts with time dynamics problems where no such limit exists.

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_NQS_neural-net_summary.png" width="1000"/></p>

<br>
There is no doubt - AI has seen some tremendous advances in the last decade or so. 


<!-- <p>
  <span style="font-size: 1.2em; line-height: 0.6;">There is no doubt</span>  - AI has seen some tremendous advances in the last decade or so.
</p> -->

Just to mention a few: large language models surpassed human performance[^1] on a multitude of tasks, [AlphaZero](https://www.nature.com/articles/s41586-020-03051-4a) humbles world’s top chess players, while [AlphaFold2&3](https://www.nature.com/articles/s41586-024-07487-w) make a huge impact on protein folding research. Personally, I find this progress in a variety of fields really quite amazing! This also naturally prompts me to ask a question about my own field: **can recent AI advancements be helpful for understanding quantum many-body physics**?

In my view the answer is **a qualified yes!** Why? Well, maybe because neural networks are *already* state-of-the-art for some quantum many-body physics models among all existing numerical methods! Neural-networks for many-body problems were shown to be highly-expressible, runtime and memory-efficient and have a different set of limitations than existing numerical methods. Cool! This makes me think that there might exist an interesting space of problems where neural nets can outcompete more traditional approaches. This in turn, will make AI-based methods to be increasingly more popular within the condensed matter / quantum information community. Motivated by this why not learning more about neural networks for quantum?

In this blogpost I will discuss how to apply **neural-network based methods for solving quantum many-body problems**. We will start from briefly describing the <a href="#neural-networks-for-quantum---basics">basic framework</a>. This should give enough background to understand the current literature on the topic. Equipped with this knowledge, we will talk about the <a href="#neural-networks-for-quantum---hopes">hopes</a> and rather unique strengths of neural networks for some quantum problems as compared with other existing methods. These will include lack of an inherent sign problem (as compared with quantum Monte Carlo) and not being limited to area law entanglement states (as compared with tensor networks). Finall, we will discuss some associated <a href="#neural-quantum-states-challenges">challenges</a> and glimpse an <a href="#outlook"> outlook </a> and perspectives of this emerging field. 

Within the blogpost I will assume you have some quantum background. I recognize though that this is an interdisciplinary field, so to make things a bit clearer for machine-learning-inclined people, please read through the extra expandable "ML boxes" to get a bit more of the quantum context. Alright, without further delay, let's get started!
* table of contents
{:toc}


## Neural networks for quantum - basics

Let's consider a problem of finding lowest energy states or time dynamics of a quantum many-body problem. Basics of applying neural networks to it are really simple. We will consider three key ideas <a href="#references">*[Carleo&Troyer (2017)]*</a>. First, we will expand a quantum state in a certain basis where coefficients will be parametrized by a neural network. Second, we will treat an expectation value of a Hamiltonian as a loss function and evaluate it through sampling. Third, we will optimize the loss function by steepest descent on neural network parameters. Note, all this is purely optimization: there is no data and we utilize neural networks only as (powerful!) function approximators[^2]. 
### Representating a quantum state

<details>
<summary><b>ML BOX 1:</b> Bra-ket notation and inner products. </summary>
<div markdown="1">
<span style="font-size:0.85em;"> $$|\psi \rangle$$ quantum state for a qubit (two-level) system denotes a vector in a tensor product Hilbert space with dimension $$2^N$$. Such vector might be decomposed in some complete basis of length $$N$$ combinations of 0’s and 1’s.  For instance $$|000\rangle = |0\rangle \otimes |0\rangle \otimes |0\rangle$$ for $$N=3$$ corresponds to a basis vector $$e_0=(1,0,0,0,0,0,0,0)$$. To denote vectors in a Hilbert space $$\mathcal{H}$$ physicists often use bra-ket notation where "ket" is denoted by $$|\psi \rangle \in \mathcal{H}$$ and dual vector "bra" by $$\langle \phi | \in \mathcal{H}^{*}$$. In such notation an inner product becomes $$\langle \phi | \psi \rangle \in \mathcal{C}$$. A quantum mechanical expectation value of an operator $$Q$$ in state $$|\psi \rangle$$ then can be written as $$\langle \psi | Q \psi \rangle$$. Throughout we assume working with an orthonormal basis $$\langle i|j\rangle = \delta_{ij}$$ thus e.g., $$\langle 000 | 001 \rangle = \langle 0 | 0 \rangle \langle 0 | 0 \rangle \langle 0 | 1 \rangle = 0$$. If you feel uncomfortable with using bra-ket notation just think of $$|\psi \rangle$$ as a $$2^N$$ dimensional complex vector $$\psi$$ (decomposable into an orthonormal complete basis as $$\psi = \sum_i \psi_i e_i$$ where $$e_i^{\dagger} e_j = \delta_{ij}$$), inner product $$\langle \phi | \psi \rangle$$ as $$\phi^{\dagger} \psi$$ where $${\dagger}$$ denotes conjugate transpose, and $$\langle \psi | Q \psi \rangle$$ as a quadratic form $$\psi^{\dagger} Q \psi$$. </span>
</div>
</details>

<br>
Let’s begin by writing a many-body quantum state on $$N$$ qubits and expand it in a complete, orthonormal basis: 
\begin{equation}
|\psi \rangle = \sum_{s} \psi_s |s \rangle
\end{equation} 
where $$\{|s\rangle\}$$ are basis vectors (e.g., in a computational basis $$|s\rangle=|100\rangle$$ for $$N=3$$) and there will be $$2^N$$ bit string elements. The key point now is to parameterize complex coefficients $$\psi_s$$ as a neural network $$\psi_s (\theta)$$ where $$\theta$$ is a vector of all neural network parameters. Such neural network takes as an input a bit string (e.g., $$s=\{-111\}$$ corresponding to the example above) and outputs a complex number $$\psi_s$$ - see the figure below. 

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_NQS_neural-net.png" width="600" loading="lazy"/></p>
Fig. 1: A simple example of a neural network for a quantum many-body problem. $$W \in \mathbb{C}^{N_{hidden}\times N}$$, $$V^T \in \mathbb{C}^{1 \times N_{hidden}}$$ are trainable matrices, $$b \in \mathbb{C}^{N_{hidden}}$$ is a trainable (bias) vector and $$\sigma(x)$$ denotes a non-linearity e.g., [RELU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). 
{:.figcaption}

Neural network parameterizes the wavefunction of a quantum system in a particular way i.e. represents it as a series of affine transformations interspersed by non-linear (activation) functions for each input bit string. For instance, for the architecture above we have $$\psi_s = V^T σ(Ws+b)$$ and $$\theta= (V,W,b)$$.

<blockquote class="note">
  <b>Key idea 1:</b> Decompose a many-body quantum state \(|\psi \rangle= \sum_{s} \psi_s |s \rangle \) and represent \(\psi_s (\theta)\) coefficients as a neural network with parameters \(\theta\). 
</blockquote>
<!-- 
After familiarizing ourselves with where to stick in a neural network, you may wonder: what is an example quantum task at hand? These might of course take many different forms. For instance one might be interested in the ground state properties of quantum systems, finite temperature states, or time dynamics of a quantum system. The above framework, often coined as neural quantum states (NQS) is applicable to all of these contexts. To illustrate the remaining key ideas we will first consider the problem of finding ground states which is conceptually the simplest. Largely similar lines of reasoning apply also to other problems - we will briefly review them later on.  -->

### Sampling quantum ground state energy
Okay, so far we have parameterized wavefunctions with a neural network. But how to solve different classes of many-body problems with it? Perhaps conceptually simplest class of problems in this context is finding lowest-energy (ground) states of a Hamiltonian and this is what we will discuss next. Solving other classes of problems (such as time dynamics or finding steady states of open systems) requires largely similar lines of reasoning - we will briefly review them later on. 

<details>
<summary><b>ML BOX 2:</b> Why searching for ground states? Why is it hard? </summary>
<div markdown="1">
<span style="font-size:0.85em;"> In a quantum many-body physics problem of $$N$$ qubits we are typically given a $$2^N \times 2^N$$ (Hamiltonian) matrix $$H$$. It turns out that many of the interesting low-temperature behavior of such system might be captured by studying the lowest-eigenvalue eigenvector of the matrix known as a "ground state". The exact approach (known as exact diagonalization) would be to numerically diagonalize the Hamiltonian matrix, however due to its exponential size (in $$N$$) it becomes quickly infeasible as $$N$$ grows beyond 20'ish qubits. This is why approximate methods for finding such eigenvectors are needed (most well-established ones largely falling into "tensor network" and "quantum Monte Carlo" methods). One paradigm for approximately finding ground states is using variational principle which states that 
\begin{equation}
\min_{\theta} \frac{\psi^{\dagger} (\theta) H \psi (\theta)}{\psi^{\dagger} \psi} \geq \lambda_{min}
\end{equation}
where $$\lambda_{min}$$ is the lowest-lying eigenvalue of $$H$$. Therefore utilizing a parametrized ansatz for a quantum state $$\psi(\theta)$$ we can find an approximation to the lowest-lying eigenvalue of it (and correspondingly to the lowest-lying eigenvector) by optimizing over parameters $$\theta$$. In this blogpost we intend to parametrize this variational ansatz with a neural network and then treat $$\frac{\psi^{\dagger} (\theta) H \psi (\theta)}{\psi^{\dagger} \psi}$$ as a loss function. 
</span>
</div>
</details>

<br>
Consider a quantum system governed by a Hamiltonian $$H$$. We will variationally approach the problem of finding a ground state of this Hamiltonian. In other words we will try to minimize an energy: $$\langle H (\theta) \rangle$$ with respect to the parameters of a neural network $$\theta$$. Let’s be a little more explicit about this:
\begin{equation}
\min_{\theta} \langle H \rangle = \min_{\theta} \frac{\psi^{\dagger} (\theta) H \psi (\theta)}{\psi^{\dagger} \psi}
\end{equation}

To proceed let's expand the above formula in an orthonormal basis (dropping $$\theta$$ for conciseness):

\begin{equation}
\langle H \rangle = \frac{\sum_{s} \sum_{s'} \psi_{s}^{\ast} \langle s | H |s' \rangle \psi_{s'}}{ \sum_r \sum_q \psi_r^{\ast} \psi_q \langle r | q \rangle} = \frac{\sum_{s} \sum_{s'} \psi_s^{\ast} H_{s s'} \psi_{s'}}{ \sum_r |\psi_r|^2}
\end{equation}
where $$H_{s s'} = \langle s | H |s' \rangle$$ are matrix elements of a Hamiltonian $$H$$.

To further proceed let’s assume[^3] that $$\psi_s \neq 0$$, we can divide and multiply by $$\psi_s$$ to get
\begin{equation}
\langle H \rangle = \sum_s p_s E_{loc}(s) 
\end{equation}
where $$p_s = \frac{|\psi_s|^2}{\sum_{s'} |\psi_{s'}|^2}$$ is a probability mass function over bit strings and $$E_{loc}(s) = \sum_{s'} H_{s s'} \frac{\psi_{s'}}{\psi_s}$$. 

**Exercise:** Convince yourself that the above derivation is correct by filling-in the missing steps by yourself!
{:.message}

Now, here comes the key practical question: can we compute $$E_{loc}(s)$$ and $$\langle H \rangle$$ efficiently? What we have done above is mostly rewriting of the original exponentially-hard problem and, indeed, both quantities in principle involve sum over *all* bit string elements - and there are still $$2^N$$ of them...

Let's consider $$E_{loc}(s)$$ first. Although, for a fully generic Hamiltonian matrix will be dense, Hamiltonians corresponding to typical physical systems will be quite sparse! In particular, for a given bit string $$s$$ (row of an exponentially large Hamiltonian matrix), there will be only polynomially many (in $$N$$) non-zero entries. This implies that summation over $$s'$$ in $$E_{loc}(s)$$ might be performed efficiently.

**Exercise:** Consider a simple form of an Ising model $$H_{ising}=\sum_{i=0}^{N-1} X_i X_{i+1}$$ in 1D with periodic boundary conditions. Convince yourself that there will be only $$\mathcal{O}(N)$$ non-zero entries per bit string element <a href="#hint1">Hint</a>.
{:.message}

That is great, but how about $$\langle H \rangle$$ evaluation? Well, utilizing the form of $$\langle H \rangle$$ we derived above, our best strategy is to evaluate a sum over exponentially many elements through sampling:
\begin{equation}
\langle H \rangle \approx \sum_{i=1}^{N_{samples}} E_{loc}(s_i) 
\end{equation}
where set of samples $$\{s_i\}$$ are typically generated by a Metropolis-Hastings algorithm[^4] and $$\{s_i\}$$ make a Monte Carlo Markov Chain (MCMC).

At first it might sound like a bit of a crazy idea! In MCMC we create a chain of bit string configurations used for sampling $$s_0 \rightarrow s_1 \rightarrow \cdots \rightarrow s_{N_{samples}}$$. If an update rule is ergodic[^5] then MCMC chain will (at least!) eventually converge to sampling from an underlying true probability distribution. Generically, it is unclear, however, how long the MCMC chains need to be in order to do so (and we know some adversarial distributions for which length of a chain, also known as *mixing time*, need to be exponentially long). So why it does not kill the method above? First, for ground states of *stoquastic* Hamiltonians it is possible to prove that the length of the MCMC chain needs to be only polynomial in system size <a href="#references">*[Bravyi+ (2023)]*</a>[^6]. Second, one can just 'hope for the best' and check some characteristics of MCMC methods (such as [autocorrelation time](http://www.hep.fsu.edu/~berg/teach/mcmc08/material/lecture07mcmc3.pdf)and [Rsplit](https://projecteuclid.org/journals/bayesian-analysis/volume-16/issue-2/Rank-Normalization-Folding-and-Localization--An-Improved-R%cb%86-for/10.1214/20-BA1221.full)) which can often tell you if something goes wrong with the sampling. Third, for some specific neural network architectures (i.e autoregressive neural networks) MCMC methods are not needed but instead one can use more reliable direct sampling <a href="#references">*[Sharir+ (2019)]*</a>.

<blockquote class="note">
  <b>Key idea 2:</b> Estimate expectation value of energy (loss function) through Monte Carlo Markov chain sampling. 
</blockquote>

### Energy optimization
Great, so we know how to evaluate energy efficiently but how to minimize it? Well, I guess the answer is quite obvious: steepest descent! 

In the simplest form it will correspond to a gradient descent algorithm for neural network parameters $$\theta$$
\begin{equation}
\theta_{t+1} = \theta_{t} - \eta \nabla_{\theta} \langle H \rangle 
\end{equation} 
where $$\eta$$ denotes learning rate. Note that the above gradient descent might be thought as stochastic gradient descent (SGD) since we evaluate gradients $$\nabla_{\theta} \langle H \rangle$$ by sampling (as discussed before). 

So why do I say steepest descent instead of just SGD? Well, in practice, for majority of architectures and models SGD performs rather poorly and it is common to use more complicated optimization methods e.g., quantum natural gradient <a href="#references">*[Stokes+ (2020)]*</a> (in the quantum Monte Carlo community also known as stochastic reconfiguration <a href="#references">*[Sorella (1998)]*</a>). The main idea of these methods is to take into account "curvature" information of the underlying parameter space manifold and therefore perform an update in a "steeper" direction than that proposed by a gradient itself. Typically such extra information is hidden in a (stochastic approximation of) matrix $$\mathbf{S}_{\alpha, \beta} = \mathbb{E} \left[ \left( \frac{\partial \log \psi_{s}(\theta)}{\partial \theta_{\alpha}} \right)^{*} \left( \frac{\partial \log \psi_{s}(\theta)}{\partial \theta_{\beta}} \right) \right] - \mathbb{E} \left[ \left( \frac{\partial \log \psi_{s}(\theta)}{\partial \theta_{\alpha}} \right)^{*} \right] \mathbb{E} \left[ \frac{\partial \log \psi_{s}(\theta)}{\partial \theta_{\beta}} \right]$$ with dimensions $$N_{parameters} \times N_{parameters}$$ (often known as quantum geometric tensor) which is said to precondition the usual gradient i.e. a quantum natural graident update is defined by
\begin{equation}
\theta_{t+1} = \theta_{t} - \eta \mathbf{S}^{-1} \nabla_{\theta} \langle H \rangle 
\end{equation}
where $$\mathbf{S}^{-1}$$ denotes (pseudo)inverse of an $$\mathbf{S}$$ matrix. 
Although I will postpone discussing $$\mathbf{S}$$ matrix (which is quite an interesting object!) in more detail for another blogpost, I will mention three important things about it from the practical perspective:

1. Quantum natural gradient is equivalent ot performing quantum imaginary time evolution on a variational manifold <a href="#references">*[Stokes+ (2020), Appendix B]*</a> 
2. Quantum natural gradient update is more costly[^7] than standard SGD due to need for pseudoinverse (increasing computation complexity to $$\mathcal{O}(N_{parameters}^3 + N_{parameters}^2 N_{samples})$$ or after information re-packaging $$\mathcal{O}(N_{samples}^3 + N_{parameters} N_{samples}^2)$$, see <a href="#references">*[Chen&Heyl (2023)]*</a>) and 
3. Matrix $$\mathbf{S}$$ is ill-conditioned which requires its regularization before it can be (pseudo-)inverted (typically in form of a diagonal shift $$\mathbf{S} \rightarrow \mathbf{S} + \epsilon \mathbb{I}$$ which decreases condition number). We will briefly go back to the quantum geometric tensor in more detail when we get to <a href="#neural-quantum-states-challenges">challenges</a> section!

<blockquote class="note">
  <b>Key idea 3:</b> Optimize the expectation value of energy through (stochastic) steepest descent. 
</blockquote>

## Neural networks for quantum - hopes

Great! So far we have studied three key ideas for applying neural networks to quantum many-body problems. To recap: 

1. Represent coefficients of a quantum state as a neural network,
2. Sample the expectation value of energy to get the loss function[^8],
3. Optimize it through steepest descent methods.

Very simple! But why is it helpful? Long story short: expressivity, efficiency inherited from ML community and no sign problem (in principle) are keywords. Let's explore these in more details!

### Expressivity of neural networks for quantum

Let's start from expressivity. Our goal is to approximate a quantum state in a Hilbert space $$\psi  \in \mathcal{H}$$ with a parametrized ansatz $$\psi(\theta) $$. One cool thing about neural network ansatze[^9] is that we are guaranteed that if the network is wide-enough we can capture any state in a Hilbert space. More specifically, even a single hidden layer neural network can approximate any quantum state with an arbitrary precision, as the number of neurons in the hidden layer goes to infinity (see figure below).

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_NQS_neural-net_UAT.png" width="600" loading="lazy"/></p>
Fig. 2: A simple sketch of a universal approximation theorem. 
{:.figcaption}

<details>
<summary><b>ML BOX 3:</b> Tensor networks and area law, quantum Monte Carlo and sign problem. </summary>
<div markdown="1">
<span style="font-size:0.85em;"> Tensor network and quantum Monte Carlo methods (both known for 20+ years) are approximate methods traditionally used within condensed matter / quantum many-body physics. <br> In tensor network methods (see e.g., [this review](https://iopscience.iop.org/article/10.1088/1751-8121/aa6dc3/pdf) for more details) one thinks of each coefficient in a quantum state basis expansion as a tensor with $$N$$ indices. In the most commonly used form (known as matrix product states or MPS) tensor with total $$2^N$$ numbers is then decomposed into a network of tensors with total $$\mathcal{O}(N \chi^2)$$ parameters (where $$\chi$$ denotes bond dimension, approximation control parameter). The catch is that such representation is practical only if $$\chi \sim \mathcal{O}(poly N)$$. This is only true for quantum states which fulfill so-called an "area" law of entanglement. This means that the value of von-Neumann entanglement entropy across any bipartition of the system only scales like an "area" of that boundary. This is somewhat confusing terminology: what is meant is that in 1D boundary is a point so entropy is bounded by a constant, in 2D boundary is a line so entropy scales with linear system dimension and finally in 3D entropy grows with a boundary of a bipartition, which corresponds to an "area" (from which the name comes from). Long-story short: whenever quantum state entanglement entropy grows quicker than area law (e.g., following "volume" law instead) then tensor networks will become much less practical (i.e. $$\chi \sim \exp(N)$$ ). Furthermore, evaluating expectation values of observables in a tensor network requires "contracting" tensor networks (think summing over indices). It turns out that such contractions can be always efficently performed for matrix product states, however for systems of larger dimensionality (2D, 3D etc.) better suited tensor network formulations, such as project entangled pair-states (PEPS) are not always efficiently contractible (which again limits their use). <br> Quantum Monte Carlo (QMC) methods (see e.g., [this book](https://www.cambridge.org/core/books/quantum-monte-carlo-approaches-for-correlated-systems/EB88C86BD9553A0738BDAE400D0B2900)) are an umbrella term for multiple methods, all commonly using Monte Carlo method to evaluate multi-dimensional integrals arising in descriptions of a many-body problem (e.g., a path integral, partition function etc.). They are not limited by dimensionality of the system, they can handle systems at finite temperature and can access large system size numerics. The catch, however, is that Monte Carlo evaluation of integrals might require an exponentially large number of samples if the problem exhibits, so-called a "sign-problem", in most cases precluding a practical use of QMC methods. This is commonly the case for fermionic Hamiltonians or non-stoquastic Hamiltonians for spins or bosons. 
</span>
</div>
</details>

<br>

Above is a cute theoretical limit but it does not sound terribly practical: $$N_{parameters} \rightarrow \infty$$ guarantee is even worse than $$2^N$$ coefficients required to naively represent any quantum state... But here comes a nice result: suppose one restricts to $$N_{parameters} \sim \mathcal{O}(poly(N))$$: how much of a Hilbert space can one represent then? Well, <a href="#references">*[Sharir+ (2020)]*</a> proved that it is strictly more than efficiently contractible[^10] tensor networks: see Fig. 3 below. 

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_NQS_neural-net_expressibility.png" width="300" loading="lazy"/></p>
Fig. 3: Expressivity of different many-body ansatze if one restricts to polynomially many parameters <a href="#references">*[Sharir+ (2020)]*</a>.
{:.figcaption}

In particular, in contrast to tensor networks, there exists **volume law states**  <a href="#references">*[Deng+ (2017)]*</a> which may be represented by simple neural network architectures (such as restricted Boltzmann machines) with only a polynomial number of parameters. This makes neural networks to be, in principle, hopeful for representing ground states of some long-range interacting Hamiltonians or quantum states arising from long quench dynamics! What is the rough intuition behind efficient representability of the volume law? 

Short answer: *non-locality* of connections within hidden units of the neural network (which contrasts with *local* connectivity of tensor networks). For instance <a href="#references">*[Deng+ (2017)]*</a> proves that for a simple example of a restricted Boltzmann machine architecture [^11], restricting connections between input layer and a hidden layer to $$\mathcal{O}(1)$$ neighbors upper bounds representable entanglement entropy to an area law. On the other hand, they prove that there exist examples of RBM states (with polynomial number of parameters) with long-range ($$\mathcal{O}(N)$$) connections which exhibit volume law - see Fig. 4 below. Thus, it is long-range connectivity which allows for efficient description of some volume law states[^12]. 

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_NQS_neural-net_volume_law.png" width="800" loading="lazy"/></p>
Fig. 4: Volume-law in restricted Boltzmann machines requires long-range connectivity.  
{:.figcaption}

This is cool: neural quantum states are strictly more expressive than tensor networks, and their representability is limited by something different than entanglement thus making it applicable to cases where generic tensor networks are not! But then a natural question becomes: what physical quantity limit neural network expressivity on quantum states? I.e. what is an equivalent of "entanglement" for tensor networks or "magic" for efficient quantum circuit simulation? Well, as of mid 2024, no one seem to know exactly and this is an exciting open research direction! 

<blockquote class="note">
  <b>Hope 1:</b> Neural quantum states are strictly more expressive than (efficiently contractible) tensor networks. 
</blockquote>

### Efficiency and ML community

Expressivity of neural quantum states is cool but it is not enough: we not only want to know that the solution of the quantum problem exists in our framework but we want to actually efficiently find it! And here comes another good thing about neural networks: due to all machine learning research and commercial incentives, neural quantum states (if deployed right!) benefit from a lot of efficiency of running ML models[^13]. What does it mean in practice? It allows to access a finite-size numerics on large system sizes, also in 2D, 3D and beyond![^14] This is again in contrast to tensor networks, which although can tackle 2D or even 3D systems, might potentially suffer from (i) contraction issues (ii) memory and runtime issues[^15]. In the literature people demonstrated these capabilities by finding state-of-the-art ground states of e.g., 40x40 square lattice Rydberg Hamiltonian  <a href="#references">*[Sprague&Czischek (2024)]*</a>  or a transverse field Ising model on a 21x21 square lattice <a href="#references">*[Sharir+ (2019)]*</a>.

<!-- Finally, a bit more vaguely: closness of the ML community makes a lot of research to be relevant to various tricks and  -->

<blockquote class="note">
  <b>Hope 2:</b> Neural quantum states are "quite" efficient both runtime and memory-wise. 
</blockquote>

### No sign problem

Just accessing large system sizes in reasonable times on lattices in 2D and 3D is not unique to neural networks: e.g., in many cases it can be done in quantum Monte Carlo methods as well. But here comes another advantage of neural quantum states: they can access regimes where quantum Monte Carlo is typically hopeless: sign-problem full Hamiltonians. For instance <a href="#references">*[Chen&Heyl (2023)]*</a> study a sign-problem-full spin J1-J2 Heisenberg model and achieve state of the art as compared with tensor network methods, and others successfuly studied fermionic systems as well (see e.g., <a href="#references">*[Moreno+ (2022)]*</a>). Is the sign problem fully gone though? Well, there is no free lunch[^16]. Achieving accurate energies on sign-problem full Hamiltonians with neural quantum states is typically more challenging than in a sign-problem free case <a href="#references">*[Szabo&Castelnovo (2020)]*</a>. The difficulty might be tracked further to appropriate representation of the phase structure of the quantum states[^17]. Although, some progress has been made in this direction by using a greater number of parameters <a href="#references">*[Chen&Heyl (2023)]*</a>, by bridging tensor networks with neural networks <a href="#references">*[Chen+ (2023)]*</a>, or pre-training on data from quantum simulators / other numerical methods <a href="#references">*[Lange+ (2024)]*</a>, properly dealing with involved phase structure of quantum amplitudes is an important open problem. 

<blockquote class="note">
  <b>Hope 3:</b> Neural quantum states do not suffer from a sign problem (at least in principle!). 
</blockquote>

## Neural quantum states: challenges

Okay, we have seen plenty of qualified hopes for neural quantum states. What are the key extra challenges? 

### Local minima issue and ground states

Well, for ground states of more complicated Hamiltonians, optimization often gets stuck in local minima (see Fig. 5). 

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_NQS_neural-net_local_minima.png" width="800" loading="lazy"/></p>
Fig. 5: Neural network optimization often gets stuck in local minima. 
{:.figcaption}

Unfortuantely, this is often true even as one ramps up the number of parameters in a neural network. This is not good! It means that in such cases we have no knob to tune for systematical improvement of accuracy... Recent work <a href="#references">*[Dash+ (2024)]*</a> links such "practical breakdown" of scalability to properties of a quantum geometric tensor i.e. matrix $$\mathbf{S}$$ we mentioned in the <a href="#energy-optimization">optimization section</a>: saturation of accuracy scalability corresponds to a saturation in the rank of the quantum geometric tensor. 

Practically, however, optimization getting trapped in local minima remains an important problem. One hopeful direction is inclusion of **symmetries for neural networks**. In traditional machine learning community such problem is quite well studied comes under umbrella term of "equivariant machine learning" (or even wider, "geometric deep learning"), and was shown to be one of the key building blocks for the most successful neural networks models to date, such as [AlphaFold2&3](https://www.nature.com/articles/s41586-024-07487-w). In the neural quantum states community, imposing symmetries has been also studied and shown to by orders of magnitude improve quality of local minima (see for instance <a href="#references">*[Vieijra+ (2020)]*</a> for an $$SU(2)$$ symmetry, or <a href="#references">*[Roth+ (2020)]*</a> for inclusion of lattice symmetries) - the more of a benefit the larger the underlying symmetry group! The key intuition for the accuracy improvement comes from an effective restriction of the size of the accessible Hilbert space with symmetries. Furthermore, as we demonstrate in our recent work <a href="#references">*[Kufel+ (2024)]*</a> it turns out that the physical symmetries do not need to be "perfect" and inclusion of only **approximate symmetries for neural networks** can lead to similar, orders of magnitude improvements in accuracy over unbiased neural network architectures[^18]!
<!-- Transfer learning? -->

<blockquote class="note">
  <b>Challenge 1:</b> Ground state search with neural quantum states is often stymied by getting trapped in local minima. 
</blockquote>

### Challenges for time dynamics

Okay, enough of problems for ground states! Let's briefly touch upon time dynamics. First a little background. It turns out, that finding time-evolution of quantum states might be found in a very similar neural quantum states paradigm which we described <a href="#neural-networks-for-quantum---basics">before</a>. We still represent wavefunction with a neural net, still evaluate observables of interest through sampling, and still optimize with quantum natural gradient, although with a slightly modified loss function. As it turns out, it yields the same updates for a quantum natural gradient as described before up to picking up an extra imaginary unit $$i$$ in front of the gradient. Next, the above approach of simulating time-dynamics with neural quantum states already has some successes e.g,. in achieving state-of-the-art (in some regimes) for simulating parameter sweep through a $$20 \times 20$$ square lattice transverse field Ising model <a href="#references">*[Schmitt+ (2022)]*</a>. Unfortunately, more broadly, time evolution with neural quantum states seems to be a more challenging problem than finding ground states. This manifests itself in e.g., inaccurate long-time observable evolution trajectories (see Fig. 6). 

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_NQS_neural-net_dynamics.png" width="300" loading="lazy"/></p>
Fig. 6: Neural networks often struggle to capture long-term dynamics of an observable $$Q$$. 
{:.figcaption}

Why does it happen? To an extent this is to be expected: time dynamics is more challenging for tensor networks as well! Specific reasons for this in neural networks include: 
1. Neccesity of complex phases in simulating dynamics (thus implying similar issues mentioned in the <a href="#no-sign-problem">sign-problem section</a>.).
2. Regularization required for ill-conditioning of quantum geometric tensor $$\mathbf{S}$$ (mentioned in the <a href="#energy-optimization">optimization section</a>) diverts the ansatz away from the real physical trajectory[^19]. 
3. Stochastic estimation of $$\mathbf{S}$$ matrix needs to be more accurate, and Monte Carlo estimations are prone to biased sampling issue <a href="#references">*[Sinibaldi+ (2023)]*</a>. This relates back to assuming $$\psi_s \neq 0 \ \forall s$$ we have made in derivation of Monte Carlo estimates <a href="#sampling-quantum-ground-state-energy">before</a>: it turns out whenever $$\psi_s \approx 0$$ while $$\partial_{\theta} \psi_s \neq 0$$, Monte Carlo estimates of $$\mathbf{S}$$ will be biased, rendering accurate real time dynamics tracking to be more difficult.

Nevertheless, improving time dynamics with neural quantum states sounds like a promising open research direction (with some exciting applications e.g., in context of benchmarking quantum simulators see e.g., <a href="#references">*[Shaw+ (2024)]*</a>)!

<blockquote class="note">
  <b>Challenge 2:</b> Time dynamics with neural quantum states requires a careful handling of sampling and regularization. 
</blockquote>

## Outlook

I hope I convinced you that neural networks for quantum many-body physics might be a promising research direction. We have discussed how many-body problems might be solved variationally by representing coefficients of the wavefunction as a neural network, sampling energy and optimizing with respect to the neural network parameters. We discussed the hopes associated with such variational ansatze such as strictly greater expressibiltiy than (efficiently contractible) tensor networks, memory and runtime efficiency and no limitation to sign-problem-free Hamiltonians. Then we talked about challenges of neural quantum states which for ground states corresponded to ansatze getting trapped in local minima. This is quite a bit of stuff - congratulations on staying with me until this point. 

Obviously, we have only scratched a surface of neural network methods for quantum many-body physics! If you would like to learn more about them see the FAQ section below with further reading suggestions and stay tuned for future blogposts on related topics!


## FAQ
1. What are typical neural network architectures for quantum many-body physics problems? Initially these were rather old-fashioned restricted Boltzmann machines and fully-connected networks. Nowadays (as of 2024) people seem to have most success with architectures incorporating symmetries in some way such as convolutional neural networks or more broadly group-convolutional neural networks. Furthemore, recently, autoregressive neural networks (such as recurrent neural networks or transformers) were shown to be also promising due to the possibility of drawing independent samples from them (and thus improving Monte Carlo estimates of observables). See <a href="#references">*[Lange+ 2024(b)]*</a> for a nice, recent, review. 
2. What are some other quantum many-body physics problems for which people applied AI methods to? There is a lot! One example is studying steady states of open quantum systems (see e.g., <a href="#references">*[Nagy&Savona 2019]*</a> ). Another example, much closer to a supervised learning setup is quantum state tomography i.e. figuring out what is a quantum state (or what values do observables defined on such quantum state take) given a set of experimental measurement snapshots. For more details see e.g., <a href="#references">*[Torlai+ (2018)]*</a> review or more recent provable advantages of ML methods for state tomography <a href="#references">*[Huang+ (2022)]*</a>. Neural quantum states may be also applied to continuous molecular systems, fermionic or bosonic models and quantum circuit simulation. For a short review on this see <a href="#references">*[Medvidović&Moreno (2024)]*</a>. If you search for even a more comprehensive, older, review on ML for quantum physics [check this](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.91.045002).
3. I am interested in playing around with neural quantum states, how do I begin? Explore [**NetKet**](https://www.netket.org/), a cool open-access neural network library for quantum many-body physics in Python! With extensive tutorials in its [documentation](https://netket.readthedocs.io/en/latest/index.html) and integration with advanced ML libraries like [JaX](https://jax.readthedocs.io/en/latest/), you will start smoothly with neural quantum states.

**Acknowledgements**: I thank Jack Kemp, DinhDuy Vu, Chris Laumann and Norm Yao for many fun discussions on neural quantum states and beyond. I also thank !!! for their helpful comments on the draft of this blogpost. Feel free to contact me at dkufel (at) g.harvard.edu if you have any questions or comments! 
{:.message}

<!--
### Header 3


```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip


### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item


### Definition lists

Name
: Godzilla

Born
: 1952

Birthplace
: Japan

Color
: Green

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this. Or is it?
```

<div class="gumroad-product-embed" data-gumroad-product-id="nuOluY"><a href="https://gumroad.com/l/nuOluY">Loading…</a></div>

## Large Tables

| Default aligned |Left aligned| Center aligned  | Right aligned  | Default aligned |Left aligned| Center aligned  | Right aligned  | Default aligned |Left aligned| Center aligned  | Right aligned  | Default aligned |Left aligned| Center aligned  | Right aligned  |
|-----------------|:-----------|:---------------:|---------------:|-----------------|:-----------|:---------------:|---------------:|-----------------|:-----------|:---------------:|---------------:|-----------------|:-----------|:---------------:|---------------:|
| First body part |Second cell | Third cell      | fourth cell    | First body part |Second cell | Third cell      | fourth cell    | First body part |Second cell | Third cell      | fourth cell    | First body part |Second cell | Third cell      | fourth cell    |
| Second line     |foo         | **strong**      | baz            | Second line     |foo         | **strong**      | baz            | Second line     |foo         | **strong**      | baz            | Second line     |foo         | **strong**      | baz            |
| Third line      |quux        | baz             | bar            | Third line      |quux        | baz             | bar            | Third line      |quux        | baz             | bar            | Third line      |quux        | baz             | bar            |
| Second body     |            |                 |                | Second body     |            |                 |                | Second body     |            |                 |                | Second body     |            |                 |                |
| 2 line          |            |                 |                | 2 line          |            |                 |                | 2 line          |            |                 |                | 2 line          |            |                 |                |
| Footer row      |            |                 |                | Footer row      |            |                 |                | Footer row      |            |                 |                | Footer row      |            |                 |                |
{:.scroll-table}


## Code blocks

~~~js
// Example can be run directly in your JavaScript console

// Create a function that takes two arguments and returns the sum of those
// arguments
var adder = new Function("a", "b", "return a + b");

// Call the function
adder(2, 6);
// > 8
~~~

 -->
## References


<a id="arjovsky">*[Arjovsky+ (2015)](https://arxiv.org/abs/1511.06464)*</a> Arjovsky, M., Shah, A. and Bengio, Y., 2016, June. Unitary evolution recurrent neural networks. In International conference on machine learning (pp. 1120-1128). PMLR. 

<a id="bravyi">*[Bravyi (2023)](https://arxiv.org/abs/2207.07044)*</a> Bravyi, S., Carleo, G., Gosset, D. and Liu, Y., 2023. A rapidly mixing Markov chain from any gapped quantum many-body system. Quantum, 7, p.1173.

<a id="carleo">*[Carleo&Troyer (2017)](https://arxiv.org/abs/1606.02318)*</a> Carleo, G. and Troyer, M., 2017. Solving the quantum many-body problem with artificial neural networks. Science, 355(6325), pp.602-606.

<a id="chen">*[Chen&Heyl (2023)](https://arxiv.org/abs/2302.01941)*</a> Chen, A. and Heyl, M., 2023. Efficient optimization of deep neural quantum states toward machine precision. arXiv preprint arXiv:2302.01941.

<a id="chen2">*[Chen+ (2023)](https://arxiv.org/abs/2304.01996)*</a> Chen, Z., Newhouse, L., Chen, E., Luo, D. and Soljacic, M., 2023. Antn: Bridging autoregressive neural networks and tensor networks for quantum many-body simulation. Advances in Neural Information Processing Systems, 36, pp.450-476.

<a id="dash">*[Dash+ (2024)](https://arxiv.org/abs/2402.01565)*</a> Dash, S., Vicentini, F., Ferrero, M. and Georges, A., 2024. Efficiency of neural quantum states in light of the quantum geometric tensor. arXiv preprint arXiv:2402.01565.

<a id="deng">*[Deng (2017)](https://arxiv.org/abs/1701.04844)*</a> Deng, D.L., Li, X. and Das Sarma, S., 2017. Quantum entanglement in neural network states. Physical Review X, 7(2), p.021021.

<a id="huang">*[Huang+ (2022)](https://arxiv.org/abs/2106.12627)*</a> Huang, H.Y., Kueng, R., Torlai, G., Albert, V.V. and Preskill, J., 2022. Provably efficient machine learning for quantum many-body problems. Science, 377(6613), p.eabk3333.

<a id="kufel">*[Kufel+ (2024)](https://arxiv.org/abs/2405.17541)*</a> Kufel, D.S., Kemp, J., Linsel, S.M., Laumann, C.R. and Yao, N.Y., 2024. Approximately-symmetric neural networks for quantum spin liquids. arXiv preprint arXiv:2405.17541.

<a id="lange">*[Lange+ (2024)](https://arxiv.org/abs/2406.00091)*</a> Lange, H., Bornet, G., Emperauger, G., Chen, C., Lahaye, T., Kienle, S., Browaeys, A. and Bohrdt, A., 2024. Transformer neural networks and quantum simulators: a hybrid approach for simulating strongly correlated systems. arXiv preprint arXiv:2406.00091.

<a id="langeb">*[Lange+ (2024)(b)](https://arxiv.org/abs/2402.09402)*</a> Lange, H., Van de Walle, A., Abedinnia, A. and Bohrdt, A., 2024. From Architectures to Applications: A Review of Neural Quantum States. arXiv preprint arXiv:2402.09402.

<a id="moreno">*[Medvidović&Moreno (2024)](https://arxiv.org/abs/2402.11014)*</a> Medvidović, M. and Moreno, J.R., 2024. Neural-network quantum states for many-body physics. arXiv prep

<a id="moreno">*[Moreno+ (2022)](https://arxiv.org/abs/2111.10420)*</a> Robledo Moreno, J., Carleo, G., Georges, A. and Stokes, J., 2022. Fermionic wave functions from neural-network constrained hidden states. Proceedings of the National Academy of Sciences, 119(32), p.e2122059119.

<a id="nagy">*[Nagy&Savona (2019)](https://arxiv.org/abs/1902.09483)*</a> Nagy, A. and Savona, V., 2019. Variational quantum Monte Carlo method with a neural-network ansatz for open quantum systems. Physical review letters, 122(25), p.250501.

<a id="roth">*[Roth+ (2023)](https://arxiv.org/abs/2211.07749)*</a> Roth, C., Szabó, A. and MacDonald, A.H., 2023. High-accuracy variational Monte Carlo for frustrated magnets with deep neural networks. Physical Review B, 108(5), p.054410.

<a id="schmitt">*[Schmitt+ (2022)](https://arxiv.org/abs/2106.09046)*</a> Schmitt, M., Rams, M.M., Dziarmaga, J., Heyl, M. and Zurek, W.H., 2022. Quantum phase transition dynamics in the two-dimensional transverse-field Ising model. Science Advances, 8(37), p.eabl6850.

<a id="sharir">*[Sharir+ (2019)](https://arxiv.org/abs/1902.04057)*</a> Sharir, O., Levine, Y., Wies, N., Carleo, G. and Shashua, A., 2020. Deep autoregressive models for the efficient variational simulation of many-body quantum systems. Physical review letters, 124(2), p.020503.

<a id="sharir">*[Sharir+ (2019b)](https://arxiv.org/abs/1803.09780)*</a> Levine, Y., Sharir, O., Cohen, N. and Shashua, A., 2019. Quantum entanglement in deep learning architectures. Physical review letters, 122(6), p.065301.

<a id="sharir">*[Sharir+ (2022)](https://arxiv.org/abs/2103.10293)*</a> Sharir, O., Shashua, A. and Carleo, G., 2022. Neural tensor contractions and the expressive power of deep neural quantum states. Physical Review B, 106(20), p.205136.

<a id="simulator">*[Shaw+ (2024)](https://arxiv.org/abs/2308.07914)*</a> Shaw, A.L., Chen, Z., Choi, J., Mark, D.K., Scholl, P., Finkelstein, R., Elben, A., Choi, S. and Endres, M., 2024. Benchmarking highly entangled states on a 60-atom analogue quantum simulator. Nature, 628(8006), pp.71-77.

<a id="sorella">*[Sorella (1998)](https://arxiv.org/abs/cond-mat/9803107)*</a> Sorella, S., 1998. Green function Monte Carlo with stochastic reconfiguration. Physical review letters, 80(20), p.4558.

<a id="sprague">*[Sprague&Czischek (2024)](https://arxiv.org/abs/2306.03921)*</a> Sprague, K. and Czischek, S., 2024. Variational Monte Carlo with large patched transformers. Communications Physics, 7(1), p.90.

<a id="stokes">*[Stokes+ (2020)](https://arxiv.org/abs/1909.02108)*</a> Stokes, J., Izaac, J., Killoran, N. and Carleo, G., 2020. Quantum natural gradient. Quantum, 4, p.269.

<a id="szabo">*[Szabo&Castelnovo (2020)](https://arxiv.org/abs/2002.04613)*</a> Szabó, A. and Castelnovo, C., 2020. Neural network wave functions and the sign problem. Physical Review Research, 2(3), p.033075.

<a id="torlai">*[Torlai+ (2018)](https://www.nature.com/articles/s41567-018-0048-5)*</a> Torlai, G., Mazzola, G., Carrasquilla, J., Troyer, M., Melko, R. and Carleo, G., 2018. Neural-network quantum state tomography. Nature Physics, 14(5), pp.447-450.

<a id="vieijra">*[Vieijra+ (2020)](https://arxiv.org/abs/1905.06034)*</a> Vieijra, T., Casert, C., Nys, J., De Neve, W., Haegeman, J., Ryckebusch, J. and Verstraete, F., 2020. Restricted Boltzmann machines for quantum states with non-Abelian or anyonic symmetries. Physical review letters, 124(9), p.097201.

## Hints to exercises
<!-- <details>
<summary>Solution</summary>
<div markdown="1">
<span style="font-size:0.85em;"> **Solution**: First note that each vertex (plaquette) operator commutes with all other vertex (plaquette) operators (since all involve purely Pauli X (or Pauli Z) operators).  Next, if plaquette operator and vertex operator do not touch one another they trivially commute (see hint). Finally even if we take plaquette operator touching vertex operator nearby, they will intersect on exactly two edges. On each edge Pauli operators anti-commute so their *product* will commute.  </span>
</div>
</details> -->

<a id="hint1">**Hint:**</a> Consider e.g., $$N=4$$ and $$ \vert 0000 \rangle$$ bit string. How many $$\langle s' \vert $$ exist such that  $$\langle s'\vert H_{ising} \vert 0000 \rangle \neq 0$$? 


<!-- <details>
<summary>Solution</summary>
<div markdown="1">
<span style="font-size:0.85em;"> **Solution**: Consider a single plaquette. The product of $$Z$$ operators around the plaquette corresponds simply to the $$B_p$$ operator. Now take two plaquettes sharing an edge. The product of $$Z$$ operators around these two plaquettes corresponds to a product of two $$B_p$$ operators (this applies $$Z$$ operator to the shared edge twice and thus is equivalent to applying no operator at all - exactly as required). You can now iterate this process to cover any set of connected plaquettes thus proving the desired statement. </span>
</div>
</details> -->