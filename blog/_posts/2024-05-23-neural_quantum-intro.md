---
layout: post
title: Neural networks for quantum many-body physics
description: >
  An intro to the neural networks for quantum many-body physics. Basic quantum background encouraged but not necessary.
sitemap: false
---

<!-- 2023-01-19 -->

<!-- related_posts:
  - /blog/_posts/2023-07-04-toric_code-lre.md -->

<!-- image: /assets/img/blog/summary.png -->

<!-- Things left to be done:
1. clearing up stuff and streamlining
2. filling in the understanding of some bits -->

*[plaquettes]: Fancy term for each of the little squares in the lattice.
*[braiding]: Loosely speaking "moving particles around each other"
*[weird]: Read: interesting
 
[^1]: This is one of the key differences between neural networks for quantum setup and typical supervised learning: we do not need to worry about generalization, only about minimizing the loss function as much as possible. 
[^2]: You might ask why it is valid to make this assumption? Well, in general it is not and for some problems this assumption is violated and introduces a bias to the sampling method! We will further discuss it later.    
[^3]: In case you are unfamiliar with Metropolis-Hastings algorithm: the key idea is to (i) start from a random bit string $$s_0$$ (ii) generate another bit string $$s_1$$ by applying an update rule to $$s_0$$ (e.g., flip a bit at a random location), (iii) calculate acceptance probability for $$s_1$$ (for simplest case of symmetric update rule as in the example above as: ) and (iv) draw a number uniformly from range $$[0,1]$$, if it is below or equal to the acceptance probability from (iii) then accept, if not then reject the $$s_1$$ configuration and draw a new bit string (v) repeat to construct a Monte Carlo Markov chain. 
<!-- $$\min(1,\frac{|\psi_{s_1}|^{2}}{|\psi_{s_0}|^{2}})$$ -->
[^4]: Ergodic in our case means that such update rule allows us to explore all bit string space.
[^5]: To fully rule out sampling inefficiency for stoquastic Hamiltonians I guess one would have to further formally show that sampling is efficient along a particular optimization pathway as well. By this I mean the following: during neural quantum state optimization one will start from a certain set of parameters $$\theta_0$$ and try converging them to the ground state with $$\theta_*$$ (provided neural network is expressive enough). Although efficient sampling of ground state with $$\theta_*$$ is then guaranteed, sampling of all intermediate states between $$\theta_0$$ and $$\theta_*$$ is not. It is not immediately clear to me however if efficient sampling everywhere along the optimization pathway is even important! 
[^6]: Note that this is not true for generic variational ansatze! 
[^7]: Just a reminder: evaluating observables on tensor networks such as MPS or PEPS requires contracting a tensor network. Although efficiently contracting MPS is not a problem, contracting PEPS is in general a #P-hard problem (think exponentially hard). 
[^8]: By this I mean few things: first existence of efficient libraries benefiting from automatic differentatiton, vectorization and just-in-time compilation (such as JAX) and second existence of hardware tailored for collecting multiple passes through a neural network in parallel such as GPUs or TPUs. 
[^9]: In 1D application of neural networks does not make so much of sense since matrix product state methods just work incredibly well! 
[^10]: Periodic boundary conditions for MPS, $$\mathcal(\chi^3)$$ complexity. 
[^11]: If the context is unfamiliar see [this](https://en.wiktionary.org/wiki/spooky_action_at_a_distance).
[^12]: Or in our case more of a 1D spatial projection of the 2D trajectory of the particle defined on a 2D toric code!
[^13]: In case this explanation was unclear or too quick: an alternative way of seeing this is the following. Imagine two $$e$$ anyons connected by a Pauli $$X$$ string. We can make them behave as an $$\epsilon$$ anyon by attaching to each $$e$$ anyon an $$m$$ particle together with a long string (which connects to a copy of an $$m$$ particles located at infinity - remember, we have shown that anyons always come in pairs!). Now exchanging two $$\epsilon$$ particles would correspond to e.g., dragging one of the $$e$$ particles (together with an $$m$$ anyon and its string!) on a semicircle while the other along the shortest line connecting original positions of two anyons. One can see that at some point in this process, the Pauli $$X$$ string attached to one of the $$m$$ anyon will necessarily cross the Pauli string $$Z$$ used to move $$e$$ anyon - and since those anti-commute, such exchange will yield an extra minus sign! If this explanation does not help either: see e.g., <a href="#references">[Simon (2020) Chp. 2 and Chp. 26.2]</a>.
[^14]: Such degeneracy, curiously comes without breaking any underlying symmetry of the Hamiltonian. If this is not surprising I would encourage you to study the symmetry-breaking paradigm due to Landau. The point of the topological order is precisely to 'disavow Landau' as [John Preskill would say](http://theory.caltech.edu/~preskill/colloquium/Balents.htm) (at least while disregarding generalization of the Landau's paradigm to so-called ["higher-form" symmetries](https://arxiv.org/pdf/2303.01817.pdf)).     
[^15]: This in principle should be considered in the thermodynamic limit (infinite number of qubits), see <a href="#references">[Bravyi et al. (2010)] p. 8</a>.
[^16]: The main idea is as follows: loosely speaking, for quantum error correction, the longer the error string the larger chance that it corresponds to an uncorrectable error. As described <a href="#anyons-and-their-statistics">before</a>, the energy cost paid for the pair of excitations of any length is the same; for self-correcting memory (for storing quantum information) we want to retain non-locality of information encoding but yet penalize for longer excitation strings. It turns out that due to their dimensionality, 3D and 4D toric code energetically penalize long excitations one of the two or both types of Pauli errors respectively.


<p style="text-align:center;"><img src="/assets/img/blog/blogpost_NQS_summary.png" width="1000"/></p>

[**IN CONSTRUCTION**]

There is no doubt - AI has seen some tremendous advances in the last decade or so. Just to mention a few: large language models surpassed human performance on a multitude of tasks (opening up doors to e.g., personalized AI tutoring), AlphaZero humbles world’s top chess players, while AlphaFold2&3 makes a huge impact on protein folding and drug discovery research. Personally, I find this progress in a variety of fields really quite amazing! This also naturally prompts me to ask a question about my own field: **can recent AI advancements be helpful for understanding quantum many-body physics**? Is the "AlphaQuantum" coming?

In my view the answer is **a qualified yes!** Why? Well, maybe because neural networks are *already* state-of-the-art for some quantum many-body physics models among all existing numerical methods[^2]! Neural-networks for many-body problems are highly-expressible, runtime and memory-efficient and have a different set of limitations than existing numerical methods. It is therefore plausible that there exists yet a larger space of problems where these methods might outcompete more traditional approaches and thus will become increasingly more popular within the condensed matter / quantum information community. Motivated by this why not learning more about neural networks for quantum? 

[classic problem](https://en.wikipedia.org/wiki/Seven_Bridges_of_K%C3%B6nigsberg) <a href="#references">*[Zeng et al. (2015)]*</a>

In this blogpost I will discuss how to apply neural-network based methods for solving quantum many-body problems. First <a href="#neural-nets-for-quantum-basics">1</a>, we will start from briefly describing the basic framework. This should give enough background to understand the current literature on the topic. Equipped with this knowledge, I will to talk about the hopes and rather unique strengths of neural networks for some quantum problems as compared with other existing methods. These will include lack of an inherent sign problem (as compared with quantum Monte Carlo) and not being limited to area law entanglement states (as compared with tensor networks). Finally, we will discuss some associated challenges and glimpse an outlook and perspectives of this emerging field. I hope it will be a fun ride! :) 

Within the blogpost I will assume you have a quantum background. I recognize though that this is an interdisciplinary field, so to make things a bit clearer for machine-learning-inclined people, please read through the extra expandable "ML boxes" to get a bit more of the context! 
* table of contents
{:toc}


## Neural networks for quantum - basics

Let's consider a problem of finding lowest energy states or time dynamics of a quantum many-body problem. Basics of applying neural networks to it are really simple. We will consider three key ideas. First, we will expand a quantum state in a certain basis where coefficients will be parametrized by a neural network. Second, we will treat an expectation value of a Hamiltonian as a loss function and evaluate it through sampling. Third, we will optimize the loss function by steepest descent on neural network parameters. Note, this is purely optimization: there is no data and utilize neural networks as (powerful!) function approximators[^1]. 

### Representating a quantum state

Let’s begin by writing a many-body quantum state on $$N$$ qubits and expand it in a complete, orthonormal basis: 
\begin{equation}
|\psi \rangle = \sum_{s} \psi_s |s \rangle
\end{equation} 
where $$\{|s\rangle\}$$ are basis vectors (e.g., in a computational basis $$|s\rangle=|100\rangle$$ for $$N=3$$) and there will be $$2^N$$ bit string elements. The key point now is to parameterize complex coefficients $$\psi_s$$ as a neural network $$\psi_s (\theta)$$ where $$\theta$$ is a vector of all neural network parameters. Such neural network takes as an input a bit string (e.g., $$s=\{-111\}$$ corresponding to the example above) and outputs a complex number $$\psi_s$$ - see the figure below. 

<p style="text-align:center;"><img src="/assets/img/blog/blogpost_NQS_neural-net.png" width="600" loading="lazy"/></p>
Fig. 1: A simple example of a neural network for a quantum many-body problem. ADD DESCRIPTION
{:.figcaption}

<details>
<summary><b>ML BOX 1:</b> Bra-ket notation and inner products. </summary>
<div markdown="1">
<span style="font-size:0.85em;"> $$|\psi \rangle$$ quantum state for a qubit (two-level) system denotes a vector in a tensor product Hilbert space with dimension $$2^N$$. Such vector might be decomposed in some complete basis of length $N$ combinations of 0’s and 1’s.  For instance $$|000\rangle = |0\rangle \otimes |0\rangle \otimes |0\rangle$$ for $$N=3$$ corresponds to a basis vector $$e_0=(1,0,0,0,0,0,0,0)$$. To denote vectors in a Hilbert space $$\mathcal{H}$$ physicists often use bra-ket notation where "ket" is denoted by $$|\psi \rangle \in \mathcal{H}$$ and dual vector "bra" by $$\langle \phi | \in \mathcal{H}^{*}$$. In such notation an inner product becomes $$\langle \phi | \psi \rangle \in \mathcal{C}$$. A quantum mechanical expectation value of an operator $$Q$$ in state $$|\psi \rangle$$ then can be written as $$\langle \psi | Q \psi \rangle$$. Throughout we assume working with an orthonormal basis $$\langle i|j\rangle = \delta_{ij}$$ thus e.g., $$\langle 000 | 001 \rangle = \langle 0 | 0 \rangle \langle 0 | 0 \rangle \langle 0 | 1 \rangle = 0$$. If you feel uncomfortable with using bra-ket notation just think of $$|\psi \rangle$$ as a $$2^N$$ dimensional complex vector $$\psi$$ (decomposable into an orthonormal complete basis as $$\psi = \sum_i \psi_i e_i$$ where $$e_i^{\dagger} e_j = \delta_{ij}$$), inner product $$\langle \phi | \psi \rangle$$ as $$\phi^{\dagger} \psi$$ where $${\dagger}$$ denotes conjugate transpose, and $$\langle \psi | Q \psi \rangle$$ as a quadratic form $$\psi^{\dagger} Q \psi$$. </span>
</div>
</details>
<!-- <span style="color:grey"><ins>D. Kufel</ins>, A. Sokal (2022)</span> -->

<br>
Neural network parameterizes the wavefunction of a quantum system in a particular way i.e. represents it as a series of affine transformations interspersed by non-linear (activation) functions for each input bit string. For instance, for the architecture above we have $$\psi_s = V^T σ(Ws+b)$$ and $$\theta= (V,W,b)$$.

<blockquote class="note">
  <b>Key idea 1:</b> Decompose a many-body quantum state \(|\psi \rangle= \sum_{s} \psi_s |s \rangle \) and represent \(\psi_s (\theta)\) coefficients as a neural network with parameters \(\theta\). 
</blockquote>
<!-- 
After familiarizing ourselves with where to stick in a neural network, you may wonder: what is an example quantum task at hand? These might of course take many different forms. For instance one might be interested in the ground state properties of quantum systems, finite temperature states, or time dynamics of a quantum system. The above framework, often coined as neural quantum states (NQS) is applicable to all of these contexts. To illustrate the remaining key ideas we will first consider the problem of finding ground states which is conceptually the simplest. Largely similar lines of reasoning apply also to other problems - we will briefly review them later on.  -->

### Sampling quantum ground state energy
Okay, so we parameterized wavefunctions with a neural network but how to solve different classes of many-body problems with it? Perhaps conceptually simplest class of problems in this context is finding lowest-energy states of a Hamiltonian and this is what we will discuss next. Solving other classes of problems (such as time dynamics or finding steady states of open systems) requires largely similar lines of reasoning - we will briefly review them later on. 

<details>
<summary><b>ML BOX 2:</b> Why searching for ground states? Why is it hard?. </summary>
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

To further proceed let’s assume[^2] that $$\psi_s \neq 0$$, we can divide and multiply by $$\psi_s$$ to get
\begin{equation}
\langle H \rangle = \sum_s p_s E_{loc}(s) 
\end{equation}
where $$p_s = \frac{|\psi_s|^2}{\sum_{s'} |\psi_{s'}|^2}$$ is a probability mass function over bit strings and $$E_{loc}(s) = \sum_{s'} H_{s s'} \frac{\psi_{s'}}{\psi_s}$$. 

Now, here comes the key practical question: can we compute $$E_{loc}(s)$$ and $$\langle H \rangle$$ efficiently? Indeed, both quantities in principle involve sum over *all* bit string elements - and there are $$2^N$$ of them. 

Let's consider $$E_{loc}(s)$$ first. Although, for a fully generic Hamiltonian matrix will be dense, Hamiltonians corresponding to typical physical systems will be quite sparse! In particular, for a given bit string $s$ (row of an exponentially large Hamiltonian matrix), there will be only polynomially many (in $$N$$) non-zero entries. This implies that summation over $$s'$$ in $$E_{loc}(s)$$ might be performed efficiently. 

That is great, but how about $$\langle H \rangle$$ evaluation? Well, utilizing the form of $$\langle H \rangle$$ we derived above, our best strategy is to evaluate a sum over exponentially many elements through sampling:
\begin{equation}
\langle H \rangle \approx \sum_{i=1}^{N_{samples}} E_{loc}(s_i) 
\end{equation}
where set of samples $$\{s_i\}$$ are typically generated by a Metropolis-Hastings algorithm[^3] and $$\{s_i\}$$ make a Monte Carlo Markov Chain (MCMC).

At first it might sound like a bit of a crazy idea! In MCMC we create a chain of bit string configurations used for sampling $$s_0 \rightarrow s_1 \rightarrow \cdots \rightarrow s_{N_{samples}}$$. If an update rule is ergodic[^4] then MCMC chain will eventually converge to sampling from an underlying true probability distribution. Generically, it is unclear, however, how long the MCMC chain need to be in order to do so (and we know some adversarial distributions for which length of a chain, also known as *mixing time*, need to be exponentially long). So why it does not kill the method? First, for ground states of *stoquastic* Hamiltonians it is possible to prove that the length of the MCMC chain needs to be only polynomial in system size \cite{gosset}[^5]. Second, one can just 'hope for the best' and check some characteristics of MCMC methods (such as autocorrelation time and Rsplit) which can often tell you if something goes wrong with the sampling. Third, for some specific neural network architectures (i.e autoregressive neural networks) MCMC methods are not needed but instead one can use more reliable direct sampling \cite{sharir}.

<blockquote class="note">
  <b>Key idea 2:</b> Estimate expectation value of energy (loss function) through Monte Carlo Markov chain sampling. 
</blockquote>

### Energy optimization
Great, so we know how to evaluate energy but how to minimize it? Well, I guess the answer is quite obvious: steepest descent! 

In the simplest form it will correspond to a gradient descent algorithm for neural network parameters $$\theta$$
\begin{equation}
\theta_{t+1} = \theta_{t} - \eta \nabla_{\theta} \langle H \rangle 
\end{equation} 
where $$\eta$$ denotes learning rate. Note that the above gradient descent might be thought as stochastic gradient descent (SGD) since we evaluate gradients $$\nabla_{\theta} \langle H \rangle$$ by sampling (as discussed before). 

So why do I say steepest descent instead of just SGD? Well, in practice, for majority of architectures and models SGD performs rather poorly and it is common to use more complicated optimization methods e.g., quantum natural gradient \cite{stokes} (also known as stochastic reconfiguration \cite{sorella}). We will discuss this in more detail when we get to challenges section!

<blockquote class="note">
  <b>Key idea 3:</b> Optimize the expectation value of energy through (stochastic) steepest descent. 
</blockquote>

## Neural networks for quantum - hopes

Great! So far we have studied three key ideas for applying neural networks to quantum many-body problems. To recap: first represent coefficients of a quantum state as a neural network, second: sample the expectation value of energy to get the loss function and third optimize it through steepest descent methods. Very simple! But why is it helpful? Long story short: expressivity, efficiency inherited from ML community and no sign problem (in principle) are keywords. 

### Expressivity of neural networks for quantum

Let's start from expressivity. Our goal is to approximate a quantum state in a Hilbert space $$\psi  \in \mathcal{H}$$ with a parametrized ansatz $$\psi(\theta) $$. One cool thing about neural network ansatze[^6] is that we are guaranteed that if the network is wide-enough we can capture any state in a Hilbert space. More specifically, even a single hidden layer neural network can approximate any quantum state with an arbitrary precision, as the number of neurons in the hidden layer goes to infinity (see figure below).

Above is a cute theoretical limit but it does not sound terribly practical: $$N_{parameters} \rightarrow \infty$$ guarantee is even worse than $$2^N$$ coefficients required to naively represent any quantum state... But here comes a nice result: suppose one restricts to $$N_{parameters} \sim \mathcal{O}(poly(N))$$: how much of a Hilbert space can one represent then? Well, \cite{sharir} proved that it is strictly more than efficiently contractible[^7] tensor networks: see Fig. !! below. 

In particular, in contrast to tensor networks, there exists **volume law states** \cite{sarma} which may be represented by simple neural network architectures (such as restricted Boltzmann machines) with only a polynomial number of parameters. This makes neural networks to be, in principle, hopeful for representing ground states of some long-range interacting Hamiltonians or quantum states arising from long quench dynamics. 

This is cool: neural quantum states are strictly more expressive than tensor networks, and their representability is limited by something different than entanglement thus making it applicable to cases where generic tensor networks are not! But then a natural question becomes: what physical quantity limit neural network expressivity on quantum states? I.e. what is an equivalent of "entanglement" for tensor networks (or "magic" for efficient quantum circuit simulation)? Well, as of mid 2024, no one knows precisely! 

<blockquote class="note">
  <b>Hope 1:</b> Neural quantum states are strictly more expressive than (efficiently contractible) tensor networks. 
</blockquote>

### Efficiency and ML community

Expressivity of neural quantum states is cool but it is not enough: we not only want to know that the solution of the quantum problem exists in our framework but we want to actually efficiently find it! And here comes another good thing about neural networks: due to all machine learning research and commercial incentives, neural quantum states (if deployed right!) benefit from a lot of efficiency of running ML models[^8]. What does it mean in practice? It allows to access a finite-size numerics on large system sizes, also in 2D, 3D and beyond![^9] This is again in contrast to tensor networks, which although can tackle 2D or even 3D systems, might suffer from (i) contraction issues (ii) memory and runtime issues[^10]. For instance, 40x40 square lattice Rydberg \cite{sprague}, TFIM on 21x21 square lattice \cite{sharir}.


<!-- Finally, a bit more vaguely: closness of the ML community makes a lot of research to be relevant to various tricks and  -->

<blockquote class="note">
  <b>Hope 2:</b> Neural quantum states are (quite) runtime and memory efficient. 
</blockquote>

### No sign problem (in principle)

Just accessing large system sizes in reasonable times on lattices in 2D and 3D is not unique to neural networks: e.g., in many cases it can be done in quantum Monte Carlo methods as well. But here comes another advantage of neural quantum states: they can access regimes where quantum Monte Carlo is hopeless: sign-problem full Hamiltonians. 

## Neural quantum states: challenges


## Conclusions

I hope I convinced you that the toric code is a model with some really cool properties. We have learnt what is the Hamiltonian describing the toric code, found its ground states (forming topologically equivalent closed loop configurations), shown how to create particle-like excitations (using Pauli strings), described their exchange statistics (which turned out to be bosonic for $$e$$ and $$m$$ and fermionic for $$\epsilon$$ anyons), and discussed how toric code ground states are locally indistinguishable (paving the way to the concept of topological order). Huuh, that is a lot - well done going through all this stuff!    

Obviously, there is still so much to say about the toric code! If this post still leaves you hungry for more stuff on the toric code (and beyond) see the FAQ section below with further reading suggestions, have a read of the [complimentary post](https://arthurpesah.me/blog/2023-05-13-surface-code/), and stay tuned for future blogposts on related topics!


## FAQ
1. Can the toric code be defined only on a square lattice? No, in principle we can define toric code on any lattice. See e.g., <a href="#references">[Simon (2020) Chp. 25.5]</a>.
2. Can the toric code be defined in the systems beyond 2D? Yes, in particular a 4D toric code is particularly interesting from the perspective of error correction since it might act as a self-correcting quantum memory[^16]. For a gentle intro from a condensed matter-ish view on toric codes beyond 2D see e.g., <a href="#references">[Savary&Balents (2016) p. 7]</a>.
3. Can the toric code be defined for systems other than qubits? Yes, for $$N$$ level system we have $$Z_N$$ toric code generalization; which we can further extend to more general groups in the so-called Kitaev double model (which can have non-Abelian anyons)! See e.g., <a href="#references">[Simon (2020) Chp. 25.6 and Chp. 29]</a>
4. Given the 4-body interaction term in the toric code Hamiltonian, can it be feasibly implemented in the lab? Yes, there are experimental demonstrations of a toric code in the lab (we will briefly discuss them in another blogpost!). If you particularly worry about the 4-body term specifically you might think of a toric code as the lowest order in perturbation theory description of the Kitaev Honeycomb model in the $$J_z \gg J_x,J_y$$ regime: see e.g., <a href="#references">[Kitaev (2006)]</a>.

**Acknowledgements**: I thank Jack Kemp, Chris Laumann and Norm Yao for many fun discussions on neural quantum states and beyond. I also thank !!! for their helpful comments on the draft of this blogpost. Feel free to contact me at dkufel (at) g.harvard.edu if you have any questions or comments! 
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
<a id="wenchenbook">*[Zeng et al. (2015)]*</a> Zeng, B., Chen, X., Zhou, D.L. and Wen, X.G., 2015. Quantum Information Meets Quantum Matter--From Quantum Entanglement to Topological Phase in Many-Body Systems. [*arXiv version*](https://arxiv.org/abs/1508.02595).

<a id="simonbook">*[Simon (2020)]*</a> Simon, S.H., 2020. Topological Quantum: Lecture Notes and Proto-Book. Unpublished prototype. [*Online Open-Access*](http://www-thphys.physics.ox.ac.uk/people/SteveSimon).

<a id="wilczek">*[Wilczek (1982)]*</a> Wilczek, F., 1982. Quantum mechanics of fractional-spin particles. Physical review letters, 49(14), p.957. [*Online Open-Access*](https://www.fuw.edu.pl/~pzdybel/WilczekAnyon.pdf)

<a id="leshouchestopo">*[Kitaev&Laumann (2010)]*</a> Kitaev, A. and Laumann, C., 2010. Topological phases and quantum computation. Exact methods in low-dimensional statistical physics and quantum computing, Lecture Notes of the Les Houches Summer School, (89), pp.101-125. [*arXiv version*](https://arxiv.org/abs/0904.2771).

<a id="stabilitytopo">*[Bravyi et al. (2010)]*</a>  Bravyi, S., Hastings, M.B. and Michalakis, S., 2010. Topological quantum order: stability under local perturbations. *Journal of mathematical physics, 51(9), p.093512* [*arXiv version*](https://arxiv.org/abs/1001.0344).

<a id="kitaevmasterpiece">*[Kitaev (2006)]*</a>  Kitaev, A., 2006. Anyons in an exactly solved model and beyond. *Annals of Physics, 321(1), pp.2-111.* [*arXiv version*](https://arxiv.org/abs/cond-mat/0506438)

<a id="spinliquids">*[Savary&Balents (2016)]*</a> Savary, L. and Balents, L., 2016. Quantum spin liquids: a review. *Reports on Progress in Physics, 80(1), p.016502*. [*arXiv version*](https://arxiv.org/abs/1601.03742)

## Hints to exercises
<a id="hint1">**Hint:**</a> Whenever Pauli X and Z meet on the same site they anti-commute, otherwise (e.g., $$[X \otimes \mathbb{1},\mathbb{1} \otimes Z]$$ they commute.
<details>
<summary>Solution</summary>
<div markdown="1">
<span style="font-size:0.85em;"> **Solution**: First note that each vertex (plaquette) operator commutes with all other vertex (plaquette) operators (since all involve purely Pauli X (or Pauli Z) operators).  Next, if plaquette operator and vertex operator do not touch one another they trivially commute (see hint). Finally even if we take plaquette operator touching vertex operator nearby, they will intersect on exactly two edges. On each edge Pauli operators anti-commute so their *product* will commute.  </span>
</div>
</details>

<a id="hint2">**Hint:**</a> Think in spirit of the discrete version of Stokes' theorem. Why should it apply here?
<details>
<summary>Solution</summary>
<div markdown="1">
<span style="font-size:0.85em;"> **Solution**: Consider a single plaquette. The product of $$Z$$ operators around the plaquette corresponds simply to the $$B_p$$ operator. Now take two plaquettes sharing an edge. The product of $$Z$$ operators around these two plaquettes corresponds to a product of two $$B_p$$ operators (this applies $$Z$$ operator to the shared edge twice and thus is equivalent to applying no operator at all - exactly as required). You can now iterate this process to cover any set of connected plaquettes thus proving the desired statement. </span>
</div>
</details>