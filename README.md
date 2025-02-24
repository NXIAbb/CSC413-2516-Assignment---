Download Link  https://programming.engineering/product/csc413-2516-assignment-4-2/


# CSC413-2516-Assignment---
CSC413/2516 Assignment 4
You will be completing this assignment with the aid of large language models (LLMs) such as ChatGPT, text-davinci-003, or code-davinci-002. To alleviate the unnecessary steps related to generating results and screenshotting, we have provided the GPT-generated solution with minimum prompting effort in ChatGPT: clauses. The goal is to help you (i) develop a solid understanding of the course materials, and (ii) gain some insight in problem-solving with LLMs. Think of this as analogous to (i) understanding the rules of addition and multiplication, and (ii) learning how to use a calculator. Note that LLMs may not be a reliable “calculator” (yet) — as you will see, GPT-like models can generate incorrect and contradicting answers. It is, therefore important that you have a good grasp of the lecture materials, so that you can evaluate the correctness of the model output, and also prompt the model toward the correct solution.

Prompt engineering. In this assignment, we ask that you try to (i) solve the problems yourself, and (ii) use LLMs to solve a selected subset of them. You will “guide” the LLMs toward desired outcomes by typing text prompts into the models. There are a number of different ways to prompt an LLM, including direct copy-pasting LATEX strings of a written question, copying function docstrings, or interactively editing the previously generated results. Prompting offers a natural and intuitive interface for humans to interact with and use LLMs. However, LLM-generated solutions depend significantly on the quality of the prompt used to steer the model, and most effective prompts come from a deep understanding of the task. You can decide how much time you want to spend as a university student vs. a prompt engineer, but we’d say it’s probably not a good idea to use more than 25% of your time on prompting LLMs. See Best Practices below for the basics of prompt engineering.

What are LLMs good for? We have divided the assignment problems into the following categories, based on our judgment of how difficult it is to obtain the correct answer using LLMs.

[Type 1] LLMs can produce almost correct answers from rather straightforward prompts, e.g., minor modification of the problem statement.

[Type 2] LLMs can produce partially correct and useful answers, but you may have to use a more sophisticated prompt (e.g., break down the problem into smaller pieces, then ask a sequence of questions), and also generate multiple times and pick the most reasonable output.

[Type 3] LLMs usually do not give the correct answer unless you try hard. This may include problems with involved mathematical reasoning or numerical computation (many GPT models do not have a built-in calculator).

[Type 4] LLMs are not suitable for the problem (e.g., graph/figure-related questions).

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

Written Assignment

What you have to submit for this part

See the top of this handout for submission directions. Here are the requirements.

The zero point questions (in black below) will not be graded, but you are more than welcome to include your answers for these as well in the submission.

For (nonzero-point) questions labeled [Type 1] [Type 2] you need to submit your own solution. Your own solution can be a copy-paste of the LLM output (if you verify that it is correct), but make sure you cite the model properly.

For (nonzero-point) questions in [Type 3] [Type 4] you only need to submit your own written solution, but we encourage you to experiment with LLMs on some of them.

For reference, here is everything you need to hand in for the first half of the PDF report a4-writeup.pdf.

Problem 1: 1.1.2[Type 1] , 1.2.1[Type 1] , 1.3.1[Type 1] , 1.3.2[Type 2]

Problem 2: 2.1.1[Type 3] , 2.1.2[Type 3] , 2.1.2[Type 1] , 2.2.1[Type 3]

Useful prompts

You could start by naively copy-pasting the question and the context as the prompt, and try to improve the generated answers by trial and error. Raw LATEX dumps are made available for the written questions to facilitate this process.

https://uoft-csc413.github.io/2023/assets/assignments/a4_raw_latex_dump.tex

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

1 RNNs and Self Attention

For any successful deep learning system, choosing the right network architecture is as important as choosing a good learning algorithm. In this question, we will explore how various architectural choices can have a significant impact on learning. We will analyze the learning performance from the perspective of vanishing /exploding gradients as they are backpropagated from the final layer to the first.

1.1 Warmup: A Single Neuron RNN

Consider an n layered fully connected network that has scalar inputs and outputs. For now, assume that all the hidden layers have a single unit, and that the weight matrices are set to 1 (because each hidden layer has a single unit, the weight matrices have a dimensionality of R1×1).

1.1.1 Effect of Activation – ReLU [0pt]

Lets say we’re using the ReLU activation. Let x be the input to the network and let f : R1 → R1 be the function the network is computing. Do the gradients necessarily have to vanish or explode as they are backpropagated? Answer this by showing that 0 ≤ |∂f∂x(x) | ≤ 1.

ChatGPT:

Assuming the ReLU activation function is used, the output of the hidden layer can be expressed as h(x) = max(0, x), where x is the input to the layer. Therefore, the output of the entire network can be expressed as f(x) = h(h(· · · (h(x)))), where there are n consecutive applications of h.

Let’s first consider the case where x ≥ 0. In this case, each application of h

has a gradient of

∂f(x)

1, so the overall gradient is simply the product of n 1’s, which is 1. Therefore, |

| = 1 when

∂x

x ≥ 0.

Now let’s consider the case where x < 0. In this case, the first application of h will output 0, and all subsequent applications of h will also output 0. Therefore, the output of the entire network will be f(x) = 0, and the gradient will be 0 as well. Therefore, |∂f∂x(x) | = 0 when x < 0.

Combining these two cases, we can conclude that 0 ≤ |∂f∂x(x) | ≤ 1 for any input x. Therefore, the gradients do not necessarily have to vanish or explode as they are backpropagated, and can be bounded between 0 and 1.

1.1.2 Effect of Activation – Different weights [0.5pt] [Type 1]

Solve the problem in 1.1.1 by assuming now the weights are not 1. You can assume that the i-th hidden layer has weight wi. Do the gradients necessarily have to vanish or explode as they are backpropagated? Answer this by deriving a similar bound as in Sec 1.1.1 for the magnitude of the gradient.

ChatGPT:

Let x be the input to the network, and let f : R1 → R1 be the function the network is computing. Let hi(x) be the output of the i-th hidden layer with weight wi, and let f(x) be the output of the entire network. Then we can write:

hi(x) = wi max(0, hi−1(x))

where h0(x) = x and hn(x) is the output of the entire network. Using the chain rule, we can write:

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang

Assignment 4

∂f(x)

∂hn(x)

∂hn(x) ∂hn−1(x)

· · ·

∂h2(x) ∂h1(x)

=

=

∂x

∂x

∂hn−1(x)

∂hn−2(x)

∂h1(x) ∂x

Using the expression for hi(x), we can compute the derivatives as follows:

∂hi(x)

= w

∂

max(0, h

(x)) = w

[max(0, h

i−1

(x))]′

∂h

(x)

i ∂h

(x)

i−1

i−1

i

i−1

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

1.3.1 Complexity of Self-Attention [0.5pt] [Type 1]

Recal from Lecture 8, the total cost for scaled dot-product attention scales quadratically with the sequence length n, i.e., O n2 . We can generalize the attention equation for any similarity function sim() to the following:

n

j=1 sim(Qi, Kj)Vj

αi = n (1.1)

j=1 sim(Qi, Kj)

where the subscript of a matrix represents the i-th row as a vector. This is equivalent to the

qT k

Softmax attention if we substitute sim(q, k) = exp(√dk ). Note that for this generalized equation to be a valid attention equation, the only constraint on sim() is that it need to be non-negative, which is true for all kernel functions k(x, y) = ϕ(x)T ϕ(y), for some feature mapping ϕ(). Show that by applying kernel functions, attention can be calculated with linear complexity (i.e., O(n)).

Hint: Sub in the kernel function for the similarity function into Eq 1.1. Group the terms based on their subscript (i.e., i and j).

ChatGPT:

To show that attention can be calculated with linear complexity using kernel functions, we will substitute a kernel function for the similarity function sim() in Equation 1.1 and group the terms based on their subscript.

Let k(x, y) be a kernel function, then we can define the similarity function sim(Qi, Kj) as:

sim(Qi, Kj) = k(Qi, Kj)

Substituting this into Equation 1.1, we get:

n

αi =

j=1 k(Qi, Kj)Vj

n

j=1 k(Qi, Kj)

Now, let’s expand k(Qi, Kj) using the kernel trick:

k(Qi, Kj) = ϕ(Qi)T ϕ(Kj)

where ϕ() is a feature mapping that maps the input to a high-dimensional space.

Substituting this in the previous equation, we get:

n

ϕ(Qi)T ϕ(Kj)Vj

αi =

j=1

n

j=1 ϕ(Qi)T ϕ(Kj)

Next, we can group the terms based on their subscripts as follows:

ϕ(Qi)T n ϕ(Kj)Vj

j=1

αi = ϕ(Qi)T n ϕ(Kj)

j=1

Now, we can precompute ϕ(Kj)Vj and ϕ(Kj) for all j, and store them in matrices M and N respectively, where Mj = ϕ(Kj)Vj and Nj = ϕ(Kj). Then, we can rewrite the equation as:

ϕ(Qi)T M

αi = ϕ(Qi)T N

This can be computed for all i with a single matrix multiplication, which has a complexity of O(n), since M and N have dimensions n × d, where d is the dimensionality of the feature space.

7

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

Therefore, we have shown that by applying kernel functions, attention can be calculated with linear complexity.

1.3.2 Linear Attention with SVD [0.5pt] [Type 2]

It has been empirically shown in Transformer models that the context mapping matrix P =

QK⊤

softmax √ often has a low rank. Show that if the rank of P is k and we already have access

dk

to the SVD of P , then it is possible to compute self-attention in O(nkd) time.

ChatGPT:

If the rank of P is k, then we can express P as a product of two matrices U ∈ Rn×k and V ⊤ ∈ Rk×n, where U and V are orthonormal matrices, i.e., U⊤U = Ik and V V ⊤ = In. Therefore, we have:

P =UV⊤

Using this factorization of P , we can rewrite the self-attention function as:

QK⊤

QK⊤ ⊤

√

dk

√

dk

Self-Attention(Q, K, V ) =

V softmax

= Q(UV ⊤)K⊤V (UV ⊤)⊤

QU(V ⊤K)⊤V U⊤V U⊤ = QU(V ⊤K)⊤(V U)⊤

Here, we used the property that U⊤V = 0 because U and V have different orthonormal bases. Note that V ⊤K has a shape of k × dk, and we can compute it once since we already have access

to the SVD of P .

Therefore, given the SVD of P , we can compute the self-attention with O(nkd) time complexity, which is linear in n and d, and proportional to the rank k of the context mapping matrix P . This is more efficient than theO(n2d) time complexity required for the standard self-attention calculation.

1.3.3 Linear Attention by Projecting [0pt]

Suppose we ignore the Softmax and scaling and let P = QK⊤ ∈ Rn×n. Assume P is rank k. Show that there exist two linear projection matrices C, D ∈ Rk ×n such that P V = Q(CK)⊤DV and the right hand side can be computed in O(nkd) time. Hint: Consider using SVD in your proof.

ChatGPT:

Since P = QK⊤, we can perform SVD on K as K = UΣV ⊤, where U ∈ Rn×n and V ∈ Rdk×dk are orthogonal matrices and Σ ∈ Rdk×dk is a diagonal matrix with singular values σ1, σ2, …, σdk .

Then, we have QK⊤ = QUΣV ⊤. Since P = QK⊤ has rank k, we know that Σ has at most k non-zero singular values. Without loss of generality, let the first k singular values be non-zero, and let Σk be the corresponding diagonal matrix of singular values.

Using the first k columns of U and V (denoted Uk and Vk, respectively), we can approximate

as K ≈ UkΣkVk⊤. Substituting this approximation into the expression for P , we have P ≈ QUkΣkVk⊤, which we can rewrite as P = Q(CK)⊤DV where C = Vk⊤, D = (UkΣk)T and V = Idk .

Now, to compute P V we can first computeCK and DV separately, each in O(nkd) time. Then, we can compute Q(CK)⊤DV in O(nkd) time. Therefore, we can compute P V in O(nkd) time as required.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

Reinforcement Learning

2.1 Bellman Equation

The Bellman equation can be seen as a fix point equation to what’s called the Bellman Operator. Given a policy π, the Bellman operators T π for V : B(S) → B(S) and T π for Q : B(S × A) → B(S × A) are defined as follows:

(T πV )(s) ≜ rπ(S) + γ

P(s′|s, a)π(a|s)V (s′)

(2.1)

(T πQ)(s, a) ≜ r(s, a) + γ

P(s′|s, a)π(a′|s′)Q(s′, a′)

(2.2)

for all s ∈ S( for V ) or all (s, a) ∈ S × A( for Q).

The Bellman operators have two important properties, 1) monotonicity and 2) γ-contraction. These properties give us many guarantees such as applying the operator repeatedly will converge to an unique and optimal solution, which is what allow us to show RL algorithms such as Q-Learning converges (under certain additional assumptions, but we won’t go over them here). In this section, we will show that the Bellman operator indeed have these two properties.

2.1.1

[0.5pt] Show that the Bellman operator (on V ) has the monotonicity property. i.e., show that for a fixed policy π, if V1, V2 ∈ B(S), and V1(s) ≤ V2(s) for all s ∈ S, then we have

TπV1 ≤ TπV2

(2.3)

ChatGPT:

To show the monotonicity property of the Bellman operator T π on V , we need to prove that if V1, V2 ∈ B(S) with V1(s) ≤ V2(s) for all s ∈ S, then (T πV1)(s) ≤ (T πV2)(s) for all s ∈ S. Expanding the definition of T π for V , we have:

(T πV1)(s) = rπ(s) + γ P (s′|s, π(s))V1(s′)

s′∈S

(T πV2)(s) = rπ(s) + γ P (s′|s, π(s))V2(s′)

s′∈S

Since V1(s) ≤ V2(s) for all s ∈ S, we have:

γ P (s′|s, π(s))V1(s′) ≤ γ P (s′|s, π(s))V2(s′)

s′∈S s′∈S

⇒ rπ(s) + γ P (s′|s, π(s))V1(s′) ≤ rπ(s) + γ P (s′|s, π(s))V2(s′)

s′∈S s′∈S

⇒ (T πV1)(s) ≤ (T πV2)(s)

Therefore, we have shown that the Bellman operator T π on V has the monotonicity property.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

2.1.2

[0.5pt] Show that the Bellman operator is a γ-contraction mapping with the supremum norm (on Q). i.e. show that for a discount factor γ and Q1, Q2 ∈ B(S × A), we have

∥Tπ(Q1) − Tπ(Q2)∥∞ ≤ γ∥Q1 − Q2∥∞

(2.4)

Recall from your math classes, the supremum norm (on Q) is as follows:

∥Q∥∞ = sup |Q(s, a)|

(2.5)

(s,a)∈S×A

Hint: for some function f, we have the following. For this question, you can think about what is P and f in our case.

P (x)f(x) ≤

|P (x)f(x)| = |P (x)| · |f(x)|

≤

P (x) · sup |f(x)|

(2.6)

x∈X

= sup |f(x)| P (x) = ∥f∥∞

x∈X

where in the last line we used the fact

P (x) = 1

ChatGPT:

To show that the Bellman operator is a γ-contraction mapping with the supremum norm, we need to prove that for any Q1, Q2 ∈ B(S × A), we have:

|TπQ1 − TπQ2|∞ ≤ γ|Q1 − Q2|∞

(2.7)

Let s and a be arbitrary elements in S and A respectively. Then we have:

(T πQ1)(s, a) − (T πQ2)(s, a) = r(s, a) + γ P(s′ | s, a)π(a′ | s′)Q1(s′, a′) − r(s, a) + γ

P(s′ | s, a)π(a′ | s

s′

s′

(2.8)

= γ

P(s′ | s, a)π(a′ | s′) Q1(s′, a′) − Q2(s′, a′)

(2.9)

s′

≤ γ

P(s′ | s, a)π(a′ | s′) Q1(s′, a′) − Q2(s′, a′)

(2.10)

s′

≤ γ|Q1 − Q2|∞

(2.11)

where we used the fact that | x ax| ≤ x |ax| in the third line and the supremum norm definition

in the last line. Taking the supremum over all (s, a) ∈ S × A yields:

|TπQ1 − TπQ2|∞ ≤ γ|Q1 − Q2|∞

(2.12)

as required. Therefore, the Bellman operator is a γ-contraction mapping with the supremum norm.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

2.1.3

[0.5pt] For this question, you may assume knowledge of the reward function r(s, a) and transition probability function p(s′|s, a), where s′ is the next state.

Give a definition of v∗(s) in terms of q∗(s, a).

Give a definition of q∗(s, a) in terms of v∗(s).

Give a definition of a∗ in terms of q∗(s, a).

Give a definition of a∗ in terms of v∗(s).

ChatGPT:

v ∗ (s) is the optimal state-value function, which is defined as the maximum expected return achievable by any policy π, starting from state s, i.e., v ∗ (s) = maxπ Eπ[Gt|st = s]. We can express v ∗ (s) in terms of q ∗ (s, a) as: v ∗ (s) = maxa q ∗ (s, a), where q ∗ (s, a) is the optimal action-value function, which is defined as the maximum expected return achievable by any policy π, starting from state s, taking action a, and then following the optimal policy thereafter, i.e.,

q ∗ (s, a) = maxπ Eπ[Gt|st = s, at = a].

q ∗ (s, a) is the optimal action-value function, which is defined as the maximum expected return achievable by any policy π, starting from state s, taking action a, and then following the

optimal policy thereafter, i.e., q ∗ (s, a) = maxπ Eπ[Gt|st = s, at = a]. We can express q ∗ (s, a) in terms of v ∗ (s) as: q ∗ (s, a) = r(s, a) + γ s′ S P(s′|s, a)v ∗ (s′), where r(s, a) is the reward function for taking action a in state s, γ is the discount factor, and P(s′|s, a) is the probability of transitioning to state s′ when taking action a in state s.

c) a∗ is the optimal action to take in state s, which maximizes the action-value function, i.e., a∗ = arg maxa q ∗ (s, a).

d) a∗ in terms of v ∗ (s) is: a∗ = arg maxa[r(s, a) + γ s′ S P(s′ |s, a)v ∗ (s′)]. This means that we can find the optimal action to take in state s by computing the action-value function for each

action and selecting the action with the highest value.

2.2 Policy gradients and black box optimization

Very often we have a function f that does not give us useful gradient information: input or output may be discrete; f may be piecewise constant, nowhere differentiable, or have pathological gradients (e.g., a discontinuous saw wave on an incline, whose gradient always points away from the global optimum); or f may be a black box that we cannot backpropagate through. For example, we may have a phone app that labels photos as cats or dogs. This situation is the default in Reinforcement Learning (RL), where we can execute the environment dynamics, but we cannot see or control their internals.

We still, however, want to optimize some score function J[f] : X → R. For example, in RL, we want to learn a policy that maximizes the non-differentiable environment reward.

When using the REINFORCE strategy, we replaced the θ optimization task with a Monte-Carlo approximation. One of the key factors for a successful REINFORCE application is the variance. The higher the variance, the more “noisy” the gradient estimates will be, which can slow down the optimization process. In this section we will derive the variance of the REINFORCE estimator for a simple toy task.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

Consider a loss function, f(˜a) which is the zero-one loss of the logistic regression output, p(a|θ). The input vector has D independent scalar features, xd. We evaluate the performance of the classifier by sampling from the output of the sigmoid µ. The loss function J(θ) can be written as:

D

µ = σ

θdxd ,

(2.13)

d=1

µ

a = 1

(2.14)

p(a|θ) = Bernoulli(µ) =

,

1 − µ

a = 0

a˜ ∼ p(a|θ),

(2.15)

1

a˜ = 1

(2.16)

f(˜a) =

,

0

a˜ = 0

J(θ) = Ea˜∼p(a|θ)[f(˜a)].

(2.17)

2.2.1 Closed form expression for REINFORCE estimator [0.5pt]

Recall from above that the expression for REINFORCE estimator is:

∇θJ[θ] = Ea˜∼p(a|θ) f(˜a)

∂

log p(a = a˜|θ)

(2.18)

∂θ

We can denote the expression inside the expectation as g[θ, x]:

∂

g[θ, a˜] = f(˜a)

log p(a = a˜|θ), a˜

∼ p(a|θ)

(2.19)

∂θ

For this question, derive a closed form for the g[θ, a˜] as a deterministic function of a˜, µ, θ, and

xd.

Hint: Substitute in the log likelihood of the Bernoulli distribution.

ChatGPT:

We can start by computing the partial derivative of the log-likelihood of the Bernoulli distribu-

tion:

∂

∂

log p(a = a˜|θ) =

[˜a log µ + (1 − a˜) log(1 − µ)]

(2.20)

∂θ

∂θ

=

a˜ ∂

µ

−

1

− a˜

∂

µ

(2.21)

µ ∂θ

1

−µ∂θ

=

a˜ − µ

∂

µ.

(2.22)

µ(1 − µ) ∂θ

Next, we can plug in the expressions for µ and

∂

µ:

∂θ

∂

g[θ, a˜] = f(˜a)

log p(a = a˜|θ)

(2.23)

∂θ

= f(˜a)

a˜ − µ

∂

µ

(2.24)

µ(1 − µ) ∂θ

D

D

a˜ − σ( D d = 1θdxd)

= f(˜a)

σ(

d = 1θ x )(1

−

σ(

d = 1θ

x

))x

σ( D d = 1θdxd)(1 − σ( D d = 1θdxd))

d d

d

d

d

(2.25)

D

= f(˜a)(˜a − σ(

d = 1θdxd))xd.

(2.26)

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang

Assignment 4

Therefore, we obtain the closed form for g[θ, a˜] as f(˜a)(˜a − σ( dD=1 θdxd))xd.

2.2.2 Variance of REINFORCE estimator [0pt]

We will derive the variance of the REINFORCE estimator above. Since the gradient is is D-dimensional, the covariance of the gradients will be D × D matrix. In this question, we will only consider the variance with respect to the first parameter, i.e. Var[[]ˆgθ, a˜]1] which scalar value corresponding to the first element in the diagonal of the covariance matrix. Derive the variance of the gradient estimator as a function of the first parameter vector: Var[[]ˆgθ, a˜]1], as a function of µ, θ, and xd.

Hint: The second moment of a Bernoulli random variable is µ(1 − µ).

2.2.3 Convergence and variance of REINFORCE estimator [0 pt]

Comment on the variance in Part 2.2.2. When do we expect learning to converge slowly in terms of the output of the logistic regression model, µ?

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

Programming Assignment

What you have to submit for this part

For reference, here is everything you need to hand in:

This is the second half of your PDF report a4-writeup.pdf. Please include the solutions to the following problems. You may choose to export gnn.ipynb, dqn.ipynb as a PDF and attach it to the first half of a4-writeup.pdf.

– Question 3: 3.1[Type 1] , 3.2[Type 2] , 3.3[Type 4] , 3.4[Type 2] , 3.5[Type 4] , 3.6[Type 4]

– Question 4: 4.1[Type 1] , 4.2[Type 2] , 4.3[Type 4] .

Your code filegnn.ipynb, dqn.ipynb

Introduction

In this assignment, you’ll get hands-on experience coding and training GCN (Graph Convolution Network) and DQN (Deep Q-learning Network), one of Reinforcement Learning methods. This assignment is divided into two parts: in the first part, you will learn how to implement the vanilla version of GCN and GAT. In the second part, you will implement and train a DQN agent to learn how to play the CartPole balancing game. It will be fun to see your model performs much better than you on the simple game :).

Setting Up

We recommend that you use Colab(https://colab.research.google.com/) for the assignment. To setup the Colab environment, just open the notebooks for each part of the assignment and make a copy in your own Google Drive account.

Deliverables

Each section is followed by a checklist of deliverables to add in the assignment writeup. To also give a better sense of our expectations for the answers to the conceptual questions, we’ve put maximum sentence limits. You will not be graded for any additional sentences.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

Graph Convolution Networks[5pt]

For this part of the assignment, you will implement the vanilla version of Graph Convolution Networks (GCN) Kipf and Welling [2016] and Graph Attention Networks (GAT) Velickovi´c et al. [2018].

Basics of GCN:

Recall from the lecture, the goal of a GCN is to learn a function of signals/features on a graph

G = (V, E), which takes as inputs:

the input features of each node, xi ∈ RF (in matrix form: X ∈ R|V |×F )

some information about the graph structure, typically the adjacency matrix A

Each convolutional layer can be written as H(l+1) = f(H(l), A), for some function f(). The f()

(l) ˆ −1/2 ˆ ˆ −1/2 (l) (l)

we are using for this assignment is in the form of f(H , A) = σ(D AD H W ), where

ˆ ˆ ˆ−1ˆ ˆ

A = A + Identity and D is diagonal node degree matrix (D A normalizes A such that all rows

˜ ˆ−1/2 ˆ ˆ−1/2

sum to one). Let A = D AD . The GCN we will implement takes two convolution layers,

˜

˜

(0)

))·W

(1)

)

Z = f(X, A) = sof tmax(A

· Dropout(ReLU(AXW

Basics of GAT:

Graph Attention Network (GAT) is a novel convolution-style neural network. It operates on graph-structured data and leverages masked self-attentional layers. In this assignment, we will implement the graph attention layer.

Dataset:

The dataset we used for this assignment is Cora Sen et al. [2008]. Cora is one of standard citation network benchmark dataset (just like MNIST dataset for computer vision tasks). It that consists of 2708 scientific publications and 5429 links. Each publication is classified into one of 7 classes. Each publication is described by a word vector (length 1433) that indicates the absence/presence of the corresponding word. This is used as the features of each node for our experiment. The task is to perform node classification (predict which class each node belongs to).

Experiments:

Open [GNN notebook link] on Colab and answer the following questions.

[1pt] Implementation of Graph Convolution Layer Complete the code for GraphConvolution() Class

[1pt] Implementation of Graph Convolution Network Complete the code for GCN() Class

[0.5pt] Train your Graph Convolution Network

After implementing the required classes, now you can train your GCN. You can play with the hyperparameters in args.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

[1.5pt] Implementation of Graph Attention Layer Complete the code for GraphAttentionLayer() Class

[0.5pt] Train your Graph Attention Network

After implementing the required classes, now you can train your GAT. You can play with the hyperparameters in args.

[0.5pt] Compare your models

Compare the evaluation results for Vanilla GCN and GAT. Comment on the discrepancy in their performance (if any) and briefly explain why you think it’s the case (in 1-2 sentences).

Deliverables

Create a section in your report called Graph Convolution Networks. Add the following:

Screenshots of your GraphConvolution, GCN implementations. Highlight the lines you’ve added.

Screenshots of your GCN training output, you can just screenshot the last 10 epochs with test set results.

Screenshots of your GraphAttentionLayer implementations. Highlight the lines you’ve added.

Screenshots of your GAT training output, you can just screenshot the last 10 epochs with test set results.

Your response to the written component of question 3.6. Your analysis should not exceed 3 sentences.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

Deep Q-Learning Network (DQN) [4pt]

In this part of the assignment, we will apply Reinforcement Learning (DQN) to tackle the CartPole Balancing game, the game that seems easy but actually quite hard. If you haven’t tried it yet, I recommend you try it first [the link]. However, the difficult game for human may be very simple to a computer.


Figure 1: Image of the CartPole Balancing game from OpenAI Gym.Brockman et al. [2016]

DQN Overview

Reinforcement learning defines an environment for the agent to perform certain actions (according to the policy) that maximize the reward at every time stamp. Essentially, our aim is to train a

agent that tries to maximize the discounted, cumulative reward Rt0 =

∞

t

−

t0

rt. Because we

t=t0

γ

assume there can be infinite time stamps, the discount factor, γ, is a constant between 0 and 1 that ensures the sum converges. It makes rewards from the uncertain far future less important for our agent than the ones in the near future.

The idea of Q-learning is that if we have a function Q∗(state, action) that outputs the maximum expected cumulative reward achievable from a given state-action pair, we could easily construct a

policy (action selection rule) that maximizes the reward:

π∗(s) = argmax Q∗(s, a)

(4.1)

a

However, we don’t know everything about the world, so we don’t have access to Q∗. But, since neural networks are universal function approximators, we can simply create one and train it to resemble Q∗. For our training update rule, we will use a fact that every Q function for some policies obeys the Bellman equation:

Qπ(s, a) = r + γQπ(s′, π(s′))

(4.2)

An intuitive explanation of the structure of the Bellman equation is as follows. Suppose that the agent has received reward rt at the current state, then the maximum discounted reward from

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

this point onward is equal to the current reward plus the maximum expected discounted reward γQ∗(st+1, at+1) from the next stage onward. The difference between the two sides of the equality is known as the temporal difference error, δ:

δ = Q(s, a)

−

(r + γ max Q(s′, a))

(4.3)

a

Our goal is the minimise this error, so that we can have a good Q function to estimate the rewards given any state-action pair.

Experiments

Open the Colab notebook link to begin: [DQN notebook link]. Read through the notebook and play around with it. More detailed instructions are given in the notebook. Have fun!

[1pt] Implementation of ϵ − greedy

Complete the function get action for the agent to select an action based on current state. We want to balance exploitation and exploration through ϵ − greedy, which is explained in the notebook. Include your code snippet in your write-up.

[1pt] Implementation of DQN training step

Complete the function train for the model to perform a single step of optimization. This is basically to construct the the temporal difference error δ and perform a standard optimizer update. Notice that there are two networks in the DQN network, policy net and target net, think about how to use these two networks to construct the loss. Include your code snippet in your write-up.

[2pt] Train your DQN Agent

After implementing the required functions, now you can train your DQN Agent, and you are suggested to tune the hyperparameters listed in the notebook. Hyperparameters are important to train a good agent. After all of these, now you can validate your model by playing the CartPole Balance game! List the hyperparameters’ value you choose, your epsilon decay rule and summarize your final results from the visualizations in a few sentences in your write-up.

Deliverables

Create a section in your report called Deep Q-learning Network. Add the following:

Screenshots of your get_action, train, epsilon decay rule implementations. Highlight the lines you’ve added.

Screenshots of your reward curve, and the end time frame of your cartpole video (to show how many seconds you can balance the cartpole).

Your response to the written component of question 4.3. Your analysis should not exceed 3 sentences.

CSC413/2516 with Professor Jimmy Ba & Professor Bo Wang Assignment 4

What you need to submit

Your code files: gnn.ipynb, dqn.ipynb.

A PDF document titled a4-writeup.pdf containing code screenshots, any experiment results or visualizations, as well as your answers to the written questions.

Further Resources

For further reading on GANs, DCGAN, GCN and DQN, the following links may be useful:

Generative Adversarial Nets (Goodfellow et al., 2014)

Deconvolution and Checkerboard Artifacts (Odena et al., 2016)

Progressive Growing of GANs (Karras et al. [2017]

Analyzing and Improving the Image Quality of StyleGAN (Karras et al. [2020])

An Introduction to GANs in Tensorflow

Generative Models Blog Post from OpenAI

Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)

Deep Reinforcement Learning: A Brief Survey (Arulkumaran et al., 2017)

References

Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional net-works. CoRR, abs/1609.02907, 2016. URL http://arxiv.org/abs/1609.02907.

Petar Velickovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`o, and Yoshua Bengio. Graph Attention Networks. International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJXMpikCZ. accepted as poster.

Prithviraj Sen, Galileo Mark Namata, Mustafa Bilgic, Lise Getoor, Brian Gallagher, and Tina Eliassi-Rad. Collective classification in network data. AI Magazine, 29(3):93–106, 2008. URL http://www.cs.iit.edu/~ml/pdfs/sen-aimag08.pdf.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017.

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Ana-lyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8110–8119, 2020.
