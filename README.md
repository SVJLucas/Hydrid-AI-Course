# Hybrid AI - Final Assessment

### Professor: Céline Hudelot
### Student: Lucas José

This repository contains the final assessment for the Hybrid AI course at École CentraleSupélec in Paris, France. The final assessment involves a review of the paper titled "Physics-Informed Deep Neural Operator Networks," available at https://arxiv.org/pdf/2207.05748.


# Paper Review: Physics-Informed Deep Neural Operator Networks


## Overview of Physics-Informed Neural Networks: Challenges and Advances

Physics-Informed Neural Networks (PINNs) have emerged as a powerful framework for solving complex and poorly-defined problems across various disciplines. By integrating physical laws into the training process, PINNs promise to overcome the limitations of traditional data-driven approaches. However, their application is primarily limited to scenarios that were explicitly covered during their training phase, hindering their generalization capabilities. This discussion delves into the evolution from PINNs to more versatile frameworks, such as Neural Operators, which include notable implementations like DeepONet and Fourier Neural Operator (FNO).

## Introduction to Neural Operators in PINNs

Neural Operators, such as DeepONet, have introduced significant advancements by addressing the curse of dimensionality, a common limitation in traditional PINN frameworks. These operators are designed to function as universal approximators for both linear and nonlinear operators, essential for solving differential equations.

### DeepONet

DeepONet consists of a dual-part neural network that operates in parallel and combines to form the model's final output, as illustrated below:


<p align="center">
  <img src="https://github.com/SVJLucas/Hydrid-AI-Course/assets/60625769/3816c901-60ab-44b5-804d-2e70b0d11b36" width="500px"/>
</p>

The first part is a branch network that takes initial conditions $v(x)$, evaluated at chosen points $\{\eta_1, \eta_2, \ldots, \eta_m\}$, resulting in $v_i = \{v_i(\eta_1), v_i(\eta_2), \ldots, v_i(\eta_m)\}$. The second component, the trunk network, evaluates the solution of the equation at specific points $\xi$ where the solution is desired, i.e., $\xi = \{x_i, y_i, t_i\}$. The final solution can be expressed as:

$$
G_\theta(v_1)(\xi) = \sum_{i=1}^p b_{ri} \cdot t_{ri} = \sum_{i=1}^p b_{ri}(v_1(\eta_1), v_1(\eta_2), \ldots, v_1(\eta_m)) \cdot t_{ri}(\xi)
$$

Following the **Generalized Universal Approximation Theorem for operators**, DeepONet can be considered a universal approximator for both linear and non-linear operators. This capability forms the basis for learning functions within the solution space. Since differential equations are framed in terms of operator functions, DeepONet provides a mathematical guarantee for approximating solutions during network optimization.

Following the success of DeepONet, subsequent works have enhanced the architecture, such as incorporating Fourier terms in the branch and trunk to approximate the periodicities of both initial conditions and solutions.

### Fourier Neural Operator (FNO)



The Fourier Neural Operator (FNO) is designed by replacing the kernel integral operator with a convolution operator defined in Fourier space. This operator processes input functions defined on a regular lattice grid and outputs the interest field on the same grid points. The network's parameters are defined and learned in Fourier space, rather than physical space, focusing on the Fourier series coefficients of the output function.



<p align="center">
  <img src="https://github.com/SVJLucas/Hydrid-AI-Course/assets/60625769/0eb6365d-caae-4c15-876e-04f87b030b50" width="500px"/>
</p>

FNO features a three-component architecture:
1. **Input Lifting**: The input function $v(x)$ is lifted to a higher-dimensional representation $h(x, 0)$ through a lifting layer, often parameterized by a linear transformation or a shallow neural network.
2. **Iterative Architecture**: The architecture formulates $h(x, 0) \rightarrow h(x, 1) \rightarrow \ldots \rightarrow h(x, L)$, where $h(x, j)$ is a sequence of functions representing the values at each layer. Each layer operates as a nonlinear operator, expressed as:
   
$$
h(x, j+1) = L_{FNO_j}\[h(x, j)\] := \sigma(W_jh(x, j) + F^{-1}\[R_j \cdot F\[h(\cdot, j)\]\](x) + c_j).
$$
   
3. **Output Projection**: The output $u(x)$ is obtained by projecting $h(x, L)$ through a local transformation operator layer, $Q$.

The paper discusses challenges in applying FNOs to complex problem domains, particularly in irregular input and output domains or unstructured meshes. Advanced feature expansion techniques adapt the FNO framework to handle non-lattice meshes and complex geometries. Additionally, Wavelet Neural Operators (WNO) and Implicit FNOs (IFNO) are introduced to enhance learning efficiency and stability. Physics-Informed FNO (PINO) integrates physics-informed settings to reduce data needs and enhance convergence, showing significant potential in fields like biological tissue modeling.


### Overview of Graph Neural Networks (GNNs) and GNNs as Operators

Graph Neural Networks (GNNs) are becoming a cornerstone in the machine learning landscape due to their ability to adeptly process and predict data represented in graph structures. The versatility of GNNs is especially pertinent for tasks requiring insights at the node, edge, or entire graph level. At the core of GNN functionality is the **Message Passing Layer (MPL)**, which updates each node using a sequence of operations that harness both local and adjacent node information. This sequence involves a **Message Step**, where nodes collect data from direct connections, an **Aggregate Step**, where data from all neighboring nodes are compiled, and an **Update Step**, which integrates the collected information to refine node states. These operations are succinctly captured by the mathematical formulation:

$$
h^{(k+1)}_i = \text{update}^{(k)} \left( h^{(k)}_i, \text{aggregate}^{(k)} \left( \{ h^{(k)}_j \mid \forall j \in N(i) \} \right) \right)
$$

Extending beyond the capabilities of traditional GNNs, **Graph Kernel Networks (GKNs)** and **Non-local Kernel Networks (NKNs)** represent sophisticated developments that incorporate integral neural operators. GKNs enhance model accuracy and continuity by simulating Green's functions, crucial for solving partial differential equations (PDEs) across iterative kernel integration layers. Conversely, NKNs focus on improving stability and scalability through non-local diffusion-reaction equations, capable of managing more extensive and complex graph-based structures. This advanced approach is represented by the equation:

$$
u(x) = \int_{\Omega} \kappa_\theta(x, y, a(x), a(y))v(y) \, dy
$$

where $\kappa_\theta$, modeled as a neural network, serves as the kernel function. These innovations, based in the  emphasize the progressive integration of mathematical principles with neural network technologies, driving forward the capabilities and applications of graph-based machine learning.

## Benchmark Evaluation

Despite many comparisons across various problems, I decided to show only one of them:

####  Darcy flow in a square domain

In the benchmark study on two-dimensional sub-surface flow through a square domain with heterogeneous permeability, we explore the capabilities of data-driven Fourier Neural Operators (FNOs), Graph Kernel Networks (GKNs), and Neural Kernel Networks (NKNs) in solving the classical problem modeled by Darcy's flow equation. This high-fidelity synthetic simulation is governed by the equation:

$$
-\nabla \cdot (K(x) \nabla u(x)) = f(x), \quad \text{where} \quad x = (x, y)
$$

subject to the Dirichlet boundary condition, $u_D(x) = 0$ for all $x$ on the boundary $\partial \Omega$. Here, $K$ represents the spatially varying conductivity field, $u$ the hydraulic head, and $f$ a source term, set to unity across the domain. This setup aims to learn and compute the solution operator that maps the conductivity field $K(x)$ to the solution field $u(x)$, leveraging the capabilities of neural operators.

The simulation domain, $\Omega = [0, 1]^2$, features a conductivity field $K(x)$ modeled as a two-valued piecewise constant function influenced by random geometry, which impacts the flow dynamics significantly. A dataset comprising $140$ samples of $K(x)$, generated from a distribution $K \sim \psi \mathcal{N} (0, (-\Delta + 9I)^{-2})$, is divided into $100$ training and $40$ test samples. The datasets differ in resolution, ranging from a coarse grid size ($\Delta x = \frac{1}{15}$) used for training, to finer grids ($\Delta x = \frac{1}{30}$ and $\Delta x = \frac{1}{60}$) used for testing, allowing for an assessment of each neural operator's generalization across varying resolutions. This experiment critically evaluates the performance of FNO, GKN, and NKN in handling resolution-dependent generalization in the complex field of subsurface fluid dynamics.

<p align="center">
  <img src="https://github.com/SVJLucas/Hydrid-AI-Course/assets/60625769/d1cbf4b3-1c2f-434f-b3cb-3808dd772cf6" width="500px"/>
</p>


In the paper, they evaluated the performance of three neural operator models—FNO, GKN, and NKN—each consisting of 16 layers, as illustrated in results in the figure above. NKN showed high fidelity to ground truth solutions, outperforming the others with a relative test error of 3.28%. In contrast, GKN and FNO experienced accuracy losses, particularly at material interfaces and across broader areas, with relative test errors of 4.71% and 9.29%, respectively. These findings underscore NKN's robustness and the other models' limitations in handling complex interfaces within the dataset.

## Conclusion

The evolution from traditional Physics-Informed Neural Networks to advanced neural operator frameworks like DeepONet, FNO, and GNNs marks a significant advancement in the field of computational science and engineering. These technologies not only enhance the accuracy and efficiency of simulations but also extend the applicability of neural networks to broader, more complex domains. As these models continue to evolve, their integration into real-world applications promises to revolutionize various scientific and industrial processes.


# My Final Opinion about the Paper and some Suggestions for Next Year

I find the topic of Physics-Informed Neural Networks (PINNs) and their related methodologies quite fascinating. This interest is particularly drawn from my familiarity with both the DeepONet and Fourier Neural Operator (FNO) methodologies. However, what sets this paper apart is not just its coverage of these two methods, but also its comprehensive review of recent improvements in these areas. It was particularly gratifying to see the inclusion of the study from **NeurIPS 2020, "Fourier features let networks learn high frequency functions in low dimensional domains"**, a paper that proposed using Fourier representations at various frequencies. This approach has been widely adopted in the Geometric Learning community and later expanded to the field of computer vision for object segmentation through point analysis, as detailed in the **META paper, "Segment Anything 2023"**. Here, the idea reappears more naturally, as many partial differential equations inherently rely on solutions within a Fourier series space. The incremental nature of deep learning is showcased by the successful application of techniques developed in disparate fields to solve a variety of problems.

Additionally, the paper introduces its main contribution, the Graph Kernel Networks (GKNs), which I found to be a surprising innovation. The concept of using a Graph Neural Network (GNN) to approximate the Green's function is ingenious. I recall from the **"Handbook of Linear Partial Differential Equations for Engineers and Scientists" by Andrei D. Polyanin (2002)**, that finding the Green's function related to a specific operator can facilitate solutions to PDEs with various non-linearities (finding the Green's function is known to be a very difficult process, but in the paper, they easily accomplish it using optimization). While the authors do not elaborate on this, their approach could be pivotal in developing **foundation models** for solving at least linearly derivative PDEs. By training a network on a sufficient number of problems, it could learn to approximate the Green's function, thereby enabling solutions to boundary value problems through integration. The concept of foundation models for differential equations has been repeatedly discussed and was recently highlighted by a new initiative from researchers at Meta, Google, and other institutions through the **Polymathic AI** project (https://polymathic-ai.org/). In a recent video from April 2024 (https://www.youtube.com/watch?v=fk2r8y5TfNY&t=2651s), Miles Cranmer discusses the concept of Foundation Models for differential equations from minutes 23:51 to 32:32. This video could be a valuable addition to next year's course slides as a suggested resource, along with the paper by Miles Cranmer on the use of GNNs as inductive bias for differential equation identification in hybrid AI contexts (combining deep learning and symbolic regression) (https://arxiv.org/abs/2006.11287), which could be included in the final assessment as one of possible readings.

