下記は、添付PDF「Provably Good Planar Mappings」を可能な限り**一字一句でMarkdown形式**に忠実に変換したものです。図・画像はファイル内にある箇所で「ここに画像あり」と記載しています。出典（変換元）： 

> 注記：PDF内の図や表は画像として組版されているため、本文中では「ここに画像あり」と記載しています。ページレイアウト（二段組等）は再現していませんが、見出し・数式・箇条書き・表はMarkdownで再現しています。

# Roi Poranne, Yaron Lipman

Weizmann Institute of Science

---

## Provably Good Planar Mappings

（出典: ）

### Latest updates

[https://dl.acm.org/doi/10.1145/2601097.2601123](https://dl.acm.org/doi/10.1145/2601097.2601123)

RESEARCH-ARTICLE

**Provably good planar mappings**

**ROI PORANNE,** Weizmann Institute of Science Israel, Rehovot, Israel.
**YARON LIPMAN,** Weizmann Institute of Science Israel, Rehovot, Israel.

Open Access Support provided by:

Weizmann Institute of Science Israel.

PDF Download
2601097.2601123.pdf
07 February 2026
Total Citations: 36
Total Downloads: 911.

Published: 27 July 2014.

ACM Transactions on Graphics (TOG), Volume 33, Issue 4 (July 2014)
DOI: 10.1145/2601097.2601123. EISSN: 1557-7368.

ACM Reference Format
Poranne, R., Lipman, Y. 2014. Provably Good Planar Mappings. ACM Trans. Graph. 33, 4, Article 76 (July 2014), 11 pages. DOI = 10.1145/2601097.2601123. [http://doi.acm.org/10.1145/2601097.2601123](http://doi.acm.org/10.1145/2601097.2601123).

**Copyright Notice**
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyright © ACM 0730-0301/14/07-ART76 $15.00. DOI: [http://doi.acm.org/10.1145/2601097.2601123](http://doi.acm.org/10.1145/2601097.2601123)

---

# Abstract

The problem of planar mapping and deformation is central in computer graphics. This paper presents a framework for adapting general, smooth, function bases for building provably good planar mappings. The term "good" in this context means the map has no fold-overs (injective), is smooth, and has low isometric or conformal distortion.

Existing methods that use mesh-based schemes are able to achieve injectivity and/or control distortion, but fail to create smooth mappings, unless they use a prohibitively large number of elements, which slows them down. Meshless methods are usually smooth by construction, yet they are not able to avoid fold-overs and/or control distortion.

Our approach constrains the linear deformation spaces induced by popular smooth basis functions, such as B-Splines, Gaussian and Thin-Plate Splines, at a set of collocation points, using specially tailored convex constraints that prevent fold-overs and high distortion at these points. Our analysis then provides the required density of collocation points and/or constraint type, which guarantees that the map is injective and meets the distortion constraints over the entire domain of interest.

We demonstrate that our method is interactive at reasonably complicated settings and compares favorably to other state-of-the-art mesh and meshless planar deformation methods.

**CR Categories:** I.3.5 [Computer Graphics]: Computational Geometry and Object Modeling—Geometric algorithms, languages, and systems;

**Keywords:** meshless deformation, bijective mappings, bounded isometric distortion, conformal distortion

Links: DL PDF

---

## 1 Introduction

Space deformation is an important tool in graphics and image processing, with applications ranging from image warping and character animation, to non-rigid registration and shape analysis. The two-dimensional case has garnered a great deal of attention in recent years, as is evident from the abundance of literature on the subject. Virtually all methods attempt to find maps that possess three key properties: smoothness, injectivity and shape preservation. Furthermore, for the purpose of warping and posing characters, the method should have interactive performance. However, there is no known method that possesses all of these properties. In this paper, we provide the theory and tools to generate maps that achieve all of these properties, including interactivity.

**Figure 1:** Our method is capable of generating smooth bijective maps with controlled distortion at interactive rates. Top row: source image. bottom row: examples of deformations.
**ここに画像あり**

Previous deformation models can be roughly divided into mesh-based and meshless models. Mesh-based maps are predominantly constructed using linear finite elements, and are inherently not smooth, but can be made to look smooth by using highly dense elements. Although the methods for creating maps with controlled distortion exist, they are time-consuming, and dense meshes prohibit their use in an interactive manner. On the other hand, meshless maps are usually defined using smooth bases and hence are smooth themselves. Yet we are unaware of any known technique that ensures their injectivity and/or bounds on their distortion.

The goal of this work is to bridge this gap between mesh and meshless methods, by providing a generic framework for making any smooth function basis suitable for deformation. This is accomplished by enabling direct control over the distortion of the Jacobian during optimization, including preservation of orientation (to avoid flips). Our method generates maps by constraining the Jacobian on a dense set of "collocation" points, using an active-set approach. We show that only a sparse subset of the collocation points needs to be active at every given moment, resulting in fast performance, while retaining the distortion and injectivity guarantees. Furthermore, we derive a precise mathematical relationship between the density of the collocation points, the maximal distortion achieved on them, and the maximal distortion achieved everywhere in the domain of interest.

ACM Transactions on Graphics, Vol. 33, No. 4, Article 76, Publication Date: July 2014
[http://doi.acm.org/10.1145/2601097.2601123](http://doi.acm.org/10.1145/2601097.2601123)

**Figure 2:** Several examples created with our method. The source of each group is on the left. Note the smooth deformations and the controlled distortion as can be visually assessed from the spheres texture.
**ここに画像あり**

---

## 2 Previous Work

There is a vast amount of previous work on planar warping and deformation, and it is impossible to provide a comprehensive list here. We therefore focus only on the previous work that is closely related to this paper.

### 2.1 Mesh-based deformations

The simplest form of mesh-based deformation is done by linearly interpolating the positions of the mesh vertices, which can cause arbitrary distortions and flips. Alexa et al. [2000] suggested controlling element distortion by individually interpolating triangles in an "as-rigid-as-possible" manner, using their polar decomposition. A generalization of this approach is made by considering the mesh in special "coordinates", which capture the local shape of the mesh with some invariance property [Sheffer and Kraevoy 2004; Sorkine et al. 2004; Xu et al. 2005]. Other methods describe mesh deformations by discretizing a relevant variational problem directly over the mesh, using a finite-element or finite-difference perspective [Terzopoulos et al. 1987; Xu et al. 2005; Igarashi et al. 2005; Sorkine and Alexa 2007; Liu et al. 2008; Chao et al. 2010]. Recently, several mesh-based methods that explicitly control injectivity and distortion were introduced [Lipman 2012; Schüller et al. 2013; Chen et al. 2013]. However, these methods suffer from poor performance when the density of the mesh is high.

### 2.2 Meshless deformations

While the optimization front in mesh based methods has been developed extensively, most meshless based approaches still use only simple linear blending of basis functions to generate deformations. Instead, the effort in this field of research was put into defining better basis functions. Free-Form Deformation [Sederberg and Parry 1986] and Thin-Plate Spline (TPS) deformation [Bookstein 1989] were some of the earlier examples. Later, generalized barycentric coordinates (BC) were designed to be shape aware: for example, Mean Value Coordinates (MVC) [Floater 2003; Ju et al. 2005]. Different aspects of BC were subsequently improved [Joshi et al. 2007; Lipman et al. 2008; Weber et al. 2012; Li et al. 2013].

Related to our work, several researchers have developed methods that attain injectivity under some conditions. Floater and Kosinka [2010] proved that, for mappings between convex domains, the MVC achieves injectivity. Similarly, Kosinka and Barton [2010] provide conditions for injective maps constructed on cube-like domains by using Warren’s BC ([Warren 1996]). Recently, Schneider et al. [2013] constructed an algorithm that, by composing a series of MVC mappings, produces a bijection, up to pixel precision. Although their method is attractive, as it only uses the MVC basis, it produces mappings that are proven to be truly injective only on a finite set of points. In contrast, our method is more general and works with different basis functions and distortion measures, and the generated maps can be shown to satisfy the injectivity and distortion bounds over the entire domain.

Several methods suggest using the meshless basis functions in a framework similar to the mesh-based deformation framework, instead of directly controlling the coefficients of the basis function. This is done by sampling points inside the domain to discretize and minimize an energy. Adams et al. [2008] used shape functions as defined in [Fries T.-P. 2003] to find an interpolation between two shapes. Similarly, Ben-Chen et al. [2009] and Weber et al. [2012] used harmonic and biharmonic coordinates, respectively, but minimized an energy only on sampled points from the skeleton or boundary of the shape. Levi et al. [2013] used the so-called Interior RBF, but instead of sampling the domain, packed it with spheres. Their energy strives to retain the shape of the spheres as much as possible.

Our approach is similar to the ones mentioned above, in that we also use basis functions and optimize over a set of points. However, the previous methods have a considerable disadvantage: they cannot guarantee a bijection or limit the distortion. We overcome this limitation by first formulating an optimization problem that also bounds the distortion on a point set, based upon [Lipman 2012]. Since this does not guarantee that the distortion outside the point set is bounded, we develop sufficient conditions for the deformation to satisfy distortion bounds on the entire domain. Using these techniques, we are able to generate smooth maps that are not only injective, but are also sure to have bounded distortion.

Lastly, we mention that in the subdivision literature, injectivity of the characteristic map needs to be verified for proving C¹ continuity of the limit surface. Several related methods for analysing the Jacobian were suggested [Reif 1995; Peters and Reif 1998; Zorin 1998].

---

## 3 Method outline

**Problem statement.** We discuss the application of "handle"-based deformation. This scenario involves a user who wishes to smoothly deform a region-of-interest (\Omega \subset \mathbb{R}^2) in the plane, e.g. part of an image or a 2D character, under an allowable amount of distortion and without fold-overs. The user drives the deformation by positioning handles inside the domain, and manipulating them in order to define positional constraints. Our algorithm will supply the map that conforms best to the handles, while not violating the distortion constraints at any point (x = (x,y)) inside (\Omega). The ideal deformation (f : \Omega \to \mathbb{R}^2) can be found as the solution to the general problem:

```latex
\[
\min_f E_{\text{pos}}(f) + \lambda E_{\text{reg}}(f) \tag{1}
\]
\[
\text{s.t. } D(f; x) \le K_{\max}, \quad \forall x \in \Omega \tag{2}
\]
```

where (E_{\text{pos}}) is the positional constraints energy, (E_{\text{reg}}) is a regularization term controlling smoothness, and (D(f; x)) is a measure of the distortion of (f) at point (x). Being infinite-dimensional with infinite number of constraints, the problem in (1)-(2) is intractable, so a simplification is required.

Given a finite collection of basis functions (\mathcal{F} = { f_i }_{i=1}^{n}), where (f_i : \Omega \to \mathbb{R}), one can construct planar maps by linear combinations of the basis functions,

```latex
\[
f(x) = (u(x), v(x))^T = \sum_{i=1}^n c_i f_i(x), \tag{3}
\]
```

where (c_i = (c^1_i, c^2_i)^T \in \mathbb{R}^{2\times 1}) are column vectors. Such a map can be represented by a matrix (c = [c_1, c_2, ..., c_n] \in \mathbb{R}^{2\times n}) containing the coefficients from eq. (3) as columns.

The bases mentioned above (and others) work very well for interpolating and approximating scalar functions in the plane, due to their regularity, approximation power and simplicity. Yet using this model as-is for building planar maps can, and often does, introduce arbitrary distortions and uncontrolled fold-overs, which renders this framework suboptimal for space warping and deformation. Nevertheless, we show how to constrain (c) in the space (\mathbb{R}^{2\times n}) to provide a mechanism for constructing planar deformations with controllable distortion and without fold-overs.

**Basis functions.** Although the framework in eq. (3) is general, and can be used in theory with any basis function of choice, we chose to experiment with three popular function bases: B-Splines, Thin-Plate Splines (TPS), and Gaussians (see Table 1 on the following page). Nevertheless, the tools developed in this paper are general, and can be used to construct injective and distortion-controlled mappings from different bases as well.

**Distortion.** The distortion of a differentiable map (f) at a point (x) is defined to be some measure of how (f) changes the metric at the vicinity of (x). Most distortion measures can be formulated using the Jacobian matrix,

```latex
\[
J_f(x) =
\begin{pmatrix}
\partial_x u(x) & \partial_y u(x) \\
\partial_x v(x) & \partial_y v(x)
\end{pmatrix},
\]
```

and more specifically, its maximal and minimal singular values, which we denote by (\Sigma(f; x)) and (\sigma(f; x)), or simply (\Sigma(x)) and (\sigma(x)) when there is no risk of confusion. These values measure the extent to which the map stretches a local neighborhood near (x).

We denote the distortion measure of (f) at (x) by (D(f, x) = D(f) = D(\Sigma(x), \sigma(x))), where the greater (D) is, the greater the distortion. When (D(x) = 1) there is no distortion at all at (x). We make use of the two common measures of distortion: isometric and conformal. Isometric distortion measures the preservation of lengths and can be computed with

```latex
\[
D_{\text{iso}}(x) = \max\{\Sigma(x), 1/\sigma(x)\}.
\]
```

When (\Sigma(x) = \sigma(x) = 1), and only then, (D_{\text{iso}}(x) = 1), which implies that (f) is close to a rigid motion in the vicinity of (x). Conformal distortion, on the other hand, measures the change in angles that is introduced by the map (f) and can be calculated with

```latex
\[
D_{\text{conf}}(x) = \Sigma(x)/\sigma(x).
\]
```

When (\Sigma(x) = \sigma(x)), (D_{\text{conf}}(x)) reaches its lowest possible value of 1. This indicates that, locally, the map behaves like a similarity transformation (rigid motion with an isotropic scale).

**Fold-overs.** A continuously differentiable map (f) is locally injective at a vicinity of a point (x) if (\det J_f(x) > 0). To guarantee local injectivity, it suffices to ensure that (\sigma(x) > 0) for all (x \in \Omega), and (\det J_f(x) > 0) for a single point (x \in \Omega) (in fact, one point in each connected component of (\Omega)). Indeed, since (\sigma(x) > 0), we know that (\det J_f(x) \ne 0), and since (\det J_f(x)) is a continuous function of (x), it cannot change sign in a connected region. Global injectivity of a (proper) differential map (f : \Omega \to \mathbb{R}^2) that is locally injective is guaranteed if the domain is simply connected and (f), restricted to the boundary, is injective.

**Collocation points and the active set method.** Our goal is to control the distortion and local injectivity of the map (f) over the domain (\Omega). To this end, we maintain a set of collocation points (Z = { z_j }_{j=1}^m \subset \Omega), where we explicitly monitor and control the distortion and injectivity over. That is, we ensure that

```latex
\[
D(z_j) \le K, \quad \sigma(z_j) > 0 \tag{4}
\]
```

for all (j = 1,\ldots,m), where (K \ge 1) is a parameter. Given these bounds on the set (Z) we provide bounds on the distortion and injectivity of (f) at all points (x \in \Omega).

To allow interactive rates, we use an active set method: The constraints are set only on a sparse subset, the active set, (Z' \subset Z). Once a certain collocation point (z) violates the desired bounds in eq. (4), it is added to the active set (Z'). Collocation points at which the distortion goes sufficiently below the desired bound are removed from the active set. See Figure 3 for an illustration. Implementation details are provided in Section 5. A similar idea was used in [Bommes et al. 2013], where the active set was termed lazy constraints.

It is possible to constrain the distortion at a collocation point (z) by utilizing the simple observation that the Jacobian matrix of (f) is linear in the variables (c),

```latex
\[
J_f(x) = (\nabla u(x), \nabla v(x))^T = \sum_{i=1}^n c_i \nabla f_i(x),
\]
```

and adapting the convexification approach of [Lipman 2012] to the meshless setting. Further details are in Section 5.

We recall that the definition of the fill distance (h(Z,\Omega)) of the collocation points in the domain is the furthest distance from (Z) that can be achieved in (\Omega), namely,

```latex
\[
h(Z,\Omega) = \max_{x \in \Omega} \min_{z \in Z} \|x - z\|. \tag{5}
\]
```

**Figure 3:** Illustration of our active set approach. As the bar bends, the distortion rises above a certain threshold, causing collocation points in the region to become active (left). These points prevent the bar from collapsing (middle). Excluding these points results in a map with singularities (right).
**ここに画像あり**

---

## Modulus of continuity

One of the key aspects of this paper is the ability to ensure that the constructed maps via eq. (3) satisfy strict requirements of distortion and injectivity. This is achieved by estimating the change in the singular values functions (\sigma(x), \Sigma(x)) of the Jacobian (J_f(x)) of the map (f). For this, the notion of the modulus of continuity becomes handy: It is a tool for measuring the rate of change of a function. Specifically, a function (g : \mathbb{R}^2 \to \mathbb{R}) is said to have a modulus of continuity (\omega), or in short, is (\omega)-continuous, if it satisfies

```latex
\[
|g(x) - g(y)| \le \omega(\|x - y\|), \quad \forall x, y \in \Omega. \tag{6}
\]
```

where (|\cdot|) denotes the Euclidean norm in (\mathbb{R}^2), and (\omega : \mathbb{R}^+ \to \mathbb{R}^+) is a continuous, strictly monotone function that vanishes at 0. Section 4 explains the computation of the modulus of continuity of the singular values functions (\sigma(x), \Sigma(x)) and describes how to use it for bounding the distortion of the map (f). We also make use of the modulus of continuity of maps (vector valued functions) (g : \mathbb{R}^2 \to \mathbb{R}^2), where similarly to the scalar case, (g) is (\omega)-continuous if

```latex
\[
\|g(x) - g(y)\| \le \omega(\|x - y\|), \quad \forall x, y \in \Omega. \tag{7}
\]
```

**Figure 4:** Several more examples created with our method. Due to the smoothness of the basis functions, our method is capable of handling pointwise constraints in a smooth and graceful manner.
**ここに画像あり**

---

## 4 Bounding the Distortion

The core of our approach lies in bounding the change in the distortion at a point as it gets further away from a collocation point (z \in Z). We observe that, for many useful function bases (\mathcal{F}), given the coefficients (c) and the domain (\Omega), one can compute a modulus (\omega = \omega_{\Sigma,\sigma}) such that the singular values functions (\Sigma(x), \sigma(x)) are (\omega)-continuous. This, in turn, allows bounding the change in the singular values.

In this section we: (i) provide the general motivation for calculating the modulus (\omega) of singular values; (ii) compute (\omega) for the collection of basis functions used in this paper; (iii) show how (\omega) can be used to bound the different distortion measures; and (iv) explore the different strategies for controlling the distortion of (f) over (\Omega).

**Why (\omega) is useful?** For example, to bound (\sigma(x)) from below at all points (x \in \Omega) we assume that we have the bound (\sigma(z) \ge \delta > 0) at all collocation points (z \in Z). Then, if (\sigma(x)) is (\omega)-continuous we have (|\sigma(x) - \sigma(z)| \le \omega(|x - z|)) and therefore in particular (\sigma(x) \ge \sigma(z) - \omega(|x - z|) \ge \delta - \omega(|x - z|)). Similarly, an upper bound to (\Sigma(x)) can be found. This is described in the following lemma:

**Lemma 1.** Let (\Sigma) and (\sigma) be (\omega)-continuous functions, and let (z \in Z) be some collocation point. Then for all points (x \in \Omega),

```latex
\[
\sigma(z) - \omega(\|x - z\|) \le \sigma(x) \le \Sigma(x) \le \Sigma(z) + \omega(\|x - z\|).
\]
```

### Computing (\omega) for different (\mathcal{F})

Using Lemma 1 requires knowing the modulus of continuity of the singular value functions (\sigma(x), \Sigma(x)) of the map (f) built using an arbitrary function basis (\mathcal{F}). Although this task might seem daunting, we show that, surprisingly enough, for 2D maps, this problem can be reduced to the easier task of calculating the modulus of continuity of the Jacobian of the map (f), or equivalently, the modulus of continuity of the gradients (\nabla u) and (\nabla v), as the following lemma asserts:

**Lemma 2.** Let (\nabla u) and (\nabla v) be (\omega)-continuous in (\Omega). Then both singular values functions (\Sigma) and (\sigma) are (2\omega)-continuous.

(The proof is given in Appendix B.)

This lemma is used to compute a modulus of continuity (\omega = \omega_{\Sigma,\sigma}) for the singular values functions of a map (f) defined via eq. (3). First, we note that

```latex
\[
\|\nabla u(x) - \nabla u(y)\| \le \sum_{i=1}^n |c^1_i| \|\nabla f_i(x) - \nabla f_i(y)\|
\le \sum_{i=1}^n |c^1_i| \omega_{\nabla f_i}(\|x-y\|)
\le |||c||| \, \omega_{\nabla \mathcal{F}}(\|x-y\|), \tag{8}
\]
```

where (\omega_{\nabla f_i}) is a modulus of continuity for the gradient of the basis function (\nabla f_i), (\omega_{\nabla \mathcal{F}}) is a modulus function satisfying (\omega_{\nabla \mathcal{F}}(t) \ge \omega_{\nabla f_i}(t)) for all (t\in\mathbb{R}^+) and all (f_i \in \mathcal{F}), and we use the matrix maximum-norm (|||c||| = \max_{\ell\in{1,2}} \sum_{i=1}^n |c^\ell_i|). Equation (8) shows that the modulus of (\nabla u) is (\omega_{\nabla u} = |||c||| , \omega_{\nabla \mathcal{F}}). Similar arguments show that (\omega_{\nabla v} = |||c||| , \omega_{\nabla \mathcal{F}}). Finally, Lemma 2 tells us that

```latex
\[
\omega = 2 |||c||| \, \omega_{\nabla \mathcal{F}}. \tag{9}
\]
```

In order to use eq. (9) to bound the change in the singular value functions, the modulus of the gradient (\omega_{\nabla \mathcal{F}}) for the function basis of interest needs to be known. In Table 1 we summarize the function bases that are used in this paper, as well as the moduli of their gradients, (\omega_{\nabla \mathcal{F}}). In Appendix A we provide the derivations of these modulus functions. Note that the gradient modulus (\omega_{\nabla \mathcal{F}}) of the TPS applies only locally to points (x, y \in \mathbb{R}^2) such that (|x-y| \le (1.25 e)^{-1} \approx 0.29). However, this is not a significant restriction, as the fill distance is always smaller in practice.

### Table 1: Function bases and the gradient modulus function.

| Basis     |                                             (f_i) | (\omega_{\nabla \mathcal{F}}(t)) |       |    |
| --------- | ------------------------------------------------: | -------------------------------- | ----- | -- |
| B-Splines | (B^{(3)}*\Delta(x - x_i) B^{(3)}*\Delta(y - y_i)) | (\dfrac{4}{3\Delta^2} t)         |       |    |
| TPS       |       (\dfrac{1}{2}(|x - x_i|^2)\ln(|x - x_i|^2)) | (t(5.8 + 5                       | \ln t | )) |
| Gaussians |                 (\exp\big(-|x-x_i|^2/(2s^2)\big)) | (t / s^2)                        |       |    |

> 注：上表中のTPSの勾配モジュラスは局所的で (|x-y| \le (1.25e)^{-1}) の範囲に適用される（詳細はAppendix A）。

---

### Bounding isometric and conformal distortion

We show below how Lemma 1 and eq. (9) can be used to provide bounds on the isometric and/or conformal distortion, assuming such bounds are enforced at a set of collocation points (Z).

We start with isometric distortion and assume that at all collocation points (z \in Z) we have (D_{\text{iso}}(z) \le K), or equivalently,

```latex
\[
\Sigma(z) \le K, \quad \sigma(z) \ge 1/K. \tag{10}
\]
```

Denote for brevity (h = h(Z,\Omega)), the fill distance of (Z) in (\Omega). Then using Lemma 1 we have for all points (x \in \Omega),

```latex
\[
D_{\text{iso}}(x) \le \max\{ K + \omega(h), \; \dfrac{1}{1/K - \omega(h)} \}. \tag{11}
\]
```

This bound holds only when (1/K > \omega(h)), which implies that (\sigma(x) > 0), which in turn guarantees the injectivity of the map. Otherwise, (D_{\text{iso}}(x)) cannot be bounded.

To bound the conformal distortion, we assume that all the collocation points (z \in Z) satisfy a conformal distortion bound:

```latex
\[
\Sigma(z) \le K \sigma(z), \quad \sigma(z) \ge \delta, \tag{12}
\]
```

where the second constraint, with some constant (\delta > 0), is used to avoid (\sigma(x) = 0), which may lead to loss of injectivity. Using Lemma 1 as above, for all (x \in \Omega),

```latex
\[
D_{\text{conf}}(x) \le K \left( \dfrac{\delta + \omega(h)}{\delta - \omega(h)} \right), \quad \sigma(x) \ge \delta - \omega(h). \tag{13}
\]
```

where, as in the isometric case, (\delta > \omega(h)) is required to hold.

---

### Controlling the distortion of (f)

The bounds in eq. (11) and eqs. (13) relate the distortion of the map (f) at all points in the domain (\Omega) to the distortion (K) enforced on the collocation points (Z) and the fill distance of the collocation points (h = h(Z,\Omega)). Using these relationships one can control the distortion of the map (f) in one of three strategies:

1. Given (Z) and the distortion bound (K) on its points, bound the maximal distortion (K_{\text{max}}) of (f) everywhere else in (\Omega).

2. Given the distortion bound (K) enforced at the points (Z) and a desired distortion bound (K_{\text{max}} > K) everywhere in (\Omega), calculate the required fill distance (h) to achieve it.

3. Given (Z) and a desired distortion bound (K_{\text{max}} > 1) everywhere in (\Omega), calculate the distortion bound (K) that should be enforced on (Z).

Strategy 1 can be accomplished directly from the bounds (11),(13). For strategy 2 we need to rearrange these equations: noting that (\omega^{-1}) also monotonically increases we get

```latex
\[
h_{\text{iso}} \le \omega^{-1}\left( \min\{ K_{\text{max}} - K,\; \dfrac{1/K - 1/K_{\text{max}}}{1} \} \right) \tag{14}
\]
```

(実際の式は本文参照)

```latex
\[
h_{\text{conf}} \le \omega^{-1}\left( \dfrac{\delta}{K_{\text{max}} - K} \dfrac{K_{\text{max}} + K}{ } \right). \tag{15}
\]
```

(実際の式は本文参照)

For strategy 3 we rearrange the bounds as follows,

```latex
\[
K_{\text{iso}} \le \min\{ K_{\text{max}} - \omega(h),\; \dfrac{1}{1/K_{\text{max}} + \omega(h)} \}. \tag{16}
\]
```

```latex
\[
K_{\text{conf}} \le K_{\text{max}} \dfrac{\delta - \omega(h)}{\delta + \omega(h)}, \quad \delta_{\text{conf}} > \omega(h). \tag{17}
\]
```

(実際の式は本文参照)

**Non-convex domains and interior distances.** It is often desirable to consider a non-convex domain (\Omega), endowed with an interior distance, and basis functions defined using this distance. The definition of the fill-distance and the modulus of continuity are changed accordingly. The analysis above can then be used as-is once the gradient modulus (\omega_{\nabla \mathcal{F}}) is available, similarly to Table 1. In this paper we only provide the modulus (\omega_{\nabla \mathcal{F}}) for the Euclidean distance-based basis functions listed in that table, leaving the analysis of other bases to future work.

We emphasize that in case the non-convex domain is endowed with the Euclidean distance, the analysis holds as-is for the basis functions from Table 1. This is due to the fact that these basis functions are defined everywhere in (\mathbb{R}^2) and the modulus of their gradients is agnostic to the shape of the domain. To generate a set of collocation points with a prescribed Euclidean fill-distance in a non-convex domain it is enough to ask that the domain satisfies the cone condition (see e.g., [Wendland 2004], Definition 3.6), and to consider all the points from a surrounding uniform grid that fall inside the domain.

**Figure 5:** Deformation of a bar using various distortion constraints using B-Splines (see Section 6 for details).
**ここに画像あり**

---

## 5 Optimization and implementation details

In this section we describe the algorithm for calculating maps of the form of eq. (3), which conform to the positional constraints prescribed by the user, and satisfy distortion and injectivity requirements. This algorithm is summarized in **Algorithm 1**. The theory in Section 4 suggests replacing the optimization problem in (1)-(2) with the following:

```latex
\[
\min_c E_{\text{pos}}(f) + \lambda E_{\text{reg}}(f)
\]
\[
\text{s.t. } D(f; z) \le K, \quad \forall z \in Z,
\]
\[
f = \sum_{i=1}^n c_i f_i. \tag{18}
\]
```

where (E_{\text{pos}}) is the energy of the positional constraints that is changed during user interaction, (E_{\text{reg}}) is a regularization energy, (D = D_{\text{iso}}) or (D = D_{\text{conf}}) is the distortion type, and (K \ge 1) is a user prescribed distortion bound. According to Section 4, for the correct choice of (K) and (Z), (f) is guaranteed to be injective and have distortion smaller than (K_{\text{max}}). In the following, Eq. (18) is formulated as a Second-Order Cone Program (SOCP), which can be solved efficiently by an interior point method.

We remark here the positional constrains energy (E_{\text{pos}}) from eq. (18) can be replaced with hard constraints. In this case however, the problem may be infeasible due to the distortion bound, regardless of how the basis functions are chosen. This can occur if, for example, the isometric distortion is required to not exceed a value of (K), but two handles are pulled apart by a factor greater than (K). In an interactive session, this means that the deformation will not update until the handles are put back in acceptable positions, which can become a nuisance to the user.

**Activation of constraints.** During interaction, eq. (18) is solved constantly as the user manipulates the handles. At each optimization step, only a fraction of the collocation points is active, so removing the rest of the collocation point will not change the result, but will greatly reduce the computation time. In the following, we devise an algorithm that utilizes this fact, where collocation points may be inserted or removed from the active-set before each step.

The algorithm should make the interaction as smooth as possible; the distortion at any deactivated collocation point should not suddenly become significantly greater than (K) at any given step. Otherwise, at the next step, the point will become active, which will cause the deformation to "jump". Therefore, we opt to insert points into the active-set when the distortion on them is slightly below (K). We assume that the collocation points are sampled on a dense rectangular grid. Before each optimization step, the distortion on each collocation point is measured, and the local maxima of the distortion are found. If a local maximum has a distortion greater than (K_{\text{high}}) for a specified (K_{\text{high}} \in [1, K]), then that point is added to the active-set for the next optimization step. If any collocation point has distortion lower than (K_{\text{low}}) where (K_{\text{low}} \in [1, K_{\text{high}}]), then that point is removed from the active-set. This ensures that the collocation points with the maximal distortion are always active, and hence all other collocation points must have distortion smaller than (K). To further stabilize the process against fast movement of the handles by the user, we may keep a small subset of equally spread collocation points always active. In our implementation we used the default values (K_{\text{high}} = 0.1 + 0.9K) and (K_{\text{low}} = 0.5 + 0.5K). See Figure 6 for examples.

During an interactive session, potentially all of the collocation points can become active at once. However, this does not occur in practice, since only points that are above a threshold and are local maxima of the distortion can be activated. Thus, only a small number of isolated points will be activated at each iteration. The only scenario in which all collocation points are activated simultaneously is when the distortion is constant everywhere when it crosses the distortion bound threshold. This scenario is extremely unlikely due to nature of the deformation energy and the bases functions used.

**Figure 6:** Active-set visualization. The yellow dots represent the positions of the activated collocation points for the deformation shown. Note that some of the points remain activated throughout to stabilize the process.
**ここに画像あり**

---

### Distortion and injectivity constraints

We explicitly constrain the points in the active-set according to eq. (10) or (12). This requires constraining the singular values of the Jacobian (J_f(z)) for all (z \in Z). We provide a new formulation to the convex second-order cone constraints described in [Lipman 2012], where the singular values of the Jacobian of the map (f) are expressed in terms of the gradients of (f) (i.e., (\nabla u) and (\nabla v)), which is compact and useful for proving Lemma 2 (see Appendix B).

We define two vectors, (J_S f(x)) and (J_A f(x)), corresponding to the similarity and anti-similarity parts of (J_f(x)), as follows

```latex
\[
J_S f(x) = \frac{\nabla u(x) + I \nabla v(x)}{2}
\]
\[
J_A f(x) = \frac{\nabla u(x) - I \nabla v(x)}{2} \tag{19}
\]
```

Here (I) is the counter-clockwise rotation (2 \times 2) matrix by (\pi/2). It can be shown (see e.g. [Lehto and Virtanen 1973], ch. I.9, p. 49) that the singular values of (J_f(x)) can then be expressed as

```latex
\[
\Sigma(x) = \|J_S f(x)\| + \|J_A f(x)\|
\]
\[
\sigma(x) = \big| \|J_S f(x)\| - \|J_A f(x)\| \big|. \tag{20}
\]
```

The requirement (10) for the isometric distortion can be written in terms of (J_S f) and (J_A f), which are linear in (c). Eq. (10) then becomes

```latex
\[
\|J_S f(x_i)\| + \|J_A f(x_i)\| \le K \tag{21}
\]
\[
\|J_S f(x_i)\| - \|J_A f(x_i)\| \ge 1/K. \tag{22}
\]
```

where eq. (21) can be transformed into convex cone constraints,

```latex
\[
\|J_S f(x_i)\| \le t_i \tag{23a}
\]
\[
\|J_A f(x_i)\| \le s_i \tag{23b}
\]
\[
t_i + s_i \le K, \tag{23c}
\]
```

where (t_i, s_i) are auxiliary variables. However, trying to apply a similar transformation to eq. (22) will result in the non-convex cone-complement constraint,

```latex
\[
\|J_S f(x_i)\| \ge r_i, \tag{24}
\]
```

for an auxiliary (r_i). Following Lipman’s [2012] approach, eq. (24) can be convexified by introducing the notion of frames. A frame is a unit vector (d_i) used to replace eq. (24) by

```latex
\[
J_S f(x_i) \cdot d_i \ge r_i. \tag{25}
\]
```

Eq. (25) is a half plane that is contained in the cone-complement of eq. (24). Using (25), we can replace (22) with

```latex
\[
J_S f(x_i) \cdot d_i - s_i \ge 1/K, \tag{26}
\]
```

noting that (r_i) is actually redundant. We also note that this constraint forces the determinant to be positive.

The optimal choice of (d_i) at a certain optimization step depends on the value of (J_S f(x_i)) at the previous step. We would like the boundary of the half plane defined by (d_i) to be as far away as possible from (J_S f(x_i)) of the previous step to allow maximum maneuverability for the next step. This is achieved by setting

```latex
\[
d_i = J_S f(x_i) / \|J_S f(x_i)\|. \tag{27}
\]
```

after each step. For the conformal distortion case we write the constraints as in [Lipman 2012] in our notation:

```latex
\[
\|J_A f(x_i)\| \le \frac{K - 1}{K + 1} \, J_S f(x_i) \cdot d_i \tag{28a}
\]
\[
\|J_A f(x_i)\| \le J_S f(x_i) \cdot d_i - \delta. \tag{28b}
\]
```

### Initialization of the frames

In [Lipman 2012], the frames had to be picked correctly to guarantee feasibility. However, here, matters are simpler. Firstly, by using soft positional constraints we ensure that a solution always exists. Although the choice of frames may not be optimal in the first iteration, it will improve in subsequent steps. Secondly, the interaction usually starts from a rest pose, so the trivial solution has the identity as the Jacobian for each collocation point, and hence satisfies any distortion bound. To include the trivial solution in the feasible set we set the frames to be (d_i = (1,0)) for all (d_i).

---

### Deformation energies and positional constraints

The energy (E_{\text{pos}}(f)) for the positional constraints in eq. (18) is defined by

```latex
\[
E_{\text{pos}}(f) = \sum_l \| f(p_l) - q_l \| = \sum_l \left\| \sum_{i=1}^n c_i f_i(p_l) - q_l \right\|. \tag{29}
\]
```

where ({p_l}*{l=1}^{n_l}) and ({q_l}*{l=1}^{n_l}) are the source and target positions of the handles. We choose this energy instead of the more common quadratic energy since it is more natural in the SOCP setting, although a quadratic energy can be used as well (by adding another cone constraint). Minimizing this energy is equivalent to minimizing,

```latex
\[
\min \sum_l r_l
\quad \text{s.t. } \left\| \sum_{i=1}^n c_i f_i(p_l) - q_l \right\| \le r_l, \; \forall l \tag{30}
\]
```

where (r_l) are auxiliary variables. Eq. (30) is an SOCP, which can be combined with the distortion constraints of the previous paragraph.

As for the regularization energy (E_{\text{reg}}), we use a combination of two common functionals: the biharmonic energy, (E_{\text{bh}}) and the ARAP energy, (E_{\text{arap}}). The biharmonic energy is defined by

```latex
\[
E_{\text{bh}}(f) = E_{\text{bh}}(u,v) = \iint_\Omega \|H_u(x)\|_F^2 + \|H_v(x)\|_F^2 \, dA, \tag{31}
\]
```

where (H_u) and (H_v) are the Hessians of (u) and (v), respectively, which is a quadratic form in (c). Once (c) is taken out of the integral, the integration can be done by numerical quadrature. The ARAP energy is defined by the standard sum,

```latex
\[
E_{\text{arap}}(f) = \sum_{s=1}^{n_s} \| J_f(r_s) - Q(r_s) \|_F^2, \tag{32}
\]
```

where ({r_s}_{s=1}^{n_s}) are a set of equally spread pre-defined points, and (Q(r_s)) is the closest rotation matrix to (J_f(r_s)). Due to the non-convexity of eq. (32), incorporating it in an optimization problem usually requires a local-global approach in order to solve it (see [Liu et al. 2008]). Using the frames, it can be seen that eq. (32) can be solved via the quadratic functional,

```latex
\[
E_{\text{arap}}(f) = \sum_{s=1}^{n_s} \big( \|J_A f(x)\|_F^2 + \|J_S f(x) - d_s\|_F^2 \big), \tag{33}
\]
```

where (d_s) is the frame at (r_s).

---

### Algorithm 1: Provably good planar mapping

**Input:**

* Set of positional constraints ({p_l}*{l=1}^{n_l}) and ({q_l}*{l=1}^{n_l})
* Set of basis functions (f_i \in \mathcal{F})
* Grid of collocation points (Z = { z_j }_{j=1}^m)
* Distortion type and bound on collocation points (K \ge 1)

**Output:** Deformation (f)

**Initialization:**

* if first step then:

  * Precompute (f_i(z)) and (\nabla f_i(z)) for all (z \in Z).
  * Set (d_i = (1,0)) for all (d_i).
  * Initialize empty active set (Z').
  * Initialize set (Z'') with farthest point samples.

* Evaluate (D(z)) for (z \in Z).

* Find the set (Z_{\max}) of local maxima of (D(z)).

* foreach (z \in Z_{\max}) such that (D(z) > K_{\text{high}}) do insert (z) to (Z').

* foreach (z \in A) such that (D(z) < K_{\text{low}}) do remove (z) from (Z').

**Optimization:**

* Solve the problem in (18) using the SOCP formulation to find (c).
* Use the constraints from eq. (23) and eq. (26) for the isometric case, and those of eq. (28) in the conformal case on the collocation points in (Z' \cup Z''). Use energies from eq. (30), (31), and/or (33).

**Postprocessing:**

* Compute (f) using (c) and (\mathcal{F}).
* Update (d_i) using eq. (27).

**Return:** Deformation (f)

---

## 6 Results

**Software implementation and timing.** We have implemented an interactive software using Algorithm 1. We used Mosek [Andersen and Andersen 1999] for solving the optimization problem, and Matlab for updating the active-set. In addition, we used an external OpenGL application to interact and show the deformation. We used a machine with Intel i7 CPU clocked at 3.5 GHz. The included video shows an interactive session using our software. Figure 7 shows timings for solving the optimization problem using Mosek as a function of the number of basis functions and the size of the active-set. For this range, which covers the results in this paper, the time complexity exhibits linear behaviour. In all of our experiments we used a (200^2) grid of collocation points during interaction, and after being satisfied with the results switched to higher grid resolutions using Strategy 2 to guarantee the bounds on the distortion.

**Figure 7:** Graphs showing the time required for the optimization as a function of the number of active collocation points and the number of basis function. Note the linear behaviour.
**ここに画像あり**

**Figure 8:** Deformation of a square using TPS. Note the fold-over that caused the subsquare in the middle to disappear in the unconstrained case, and compare to the bounded isometric distortion results on the right.
**ここに画像あり**

**Parameters and function bases.** Our approach is quite versatile as the different function bases, and the distortion type and bound already attain a large variety of different results. We present a set of examples that we believe should advocate the use of our approach.

In Figure 5 we show an example of a deformation of a bar using a (6 \times 6) tensor product of uniform cubic B-Splines using the energy (E_{\text{reg}} = E_{\text{pos}} + 10^{-2} E_{\text{arap}}). Note that for the lower values of (K), the positional constraints cannot be satisfied. Also note that with no distortion constraints, the deformation creates two singularities, which were unintended and undesired. Using strategy 3 we found that in order to achieve injectivity for all cases, it was enough to check the distortion on a grid of size (3000^2). For this grid we found that for (K = 2,3,4), the maximal distortion was guaranteed to be smaller than (3.2, 10) and (49) in the isometric case respectively, and (14, 35) and (33) in the conformal case.

In Figure 8 we show additional examples of deformations of a square to demonstrate the effectiveness of our method for warping. Using TPS this time, with 25 bases positioned on a grid, we rotated two points in the middle while keeping some points on the boundary fixed. We used the smoothness energy (E_{\text{pos}} + 10^{-1} E_{\text{bh}}). In this case, the unconstrained map resulted in a fold-over that made the sub-square in the middle completely disappear, while the constrained maps stayed bijective. The required grid size that provides the injectivity certification for (K = 5) in this example was slightly less than (6000^2). For this resolution, the computation shows that for (K = 3), the maximal distortion everywhere is smaller than 7.

**Mesh-based vs. Meshless.** We compared our results with the results of previous similar mesh-based methods. In Figure 9 we show a bird image deformed with a variant of the ARAP method of Igarashi et al. [2005] as implemented in Adobe Photoshop. We compare this result to the meshless approach using ARAP energy with and without the distortion constraints. One of the main difficulties with mesh-based ARAP, which can be seen in Figure 9, is that when the object is forced to undergo a deformation that is not close to being locally rigid, cusps with fold-overs appear near the handles. This cannot happen when the basis functions are smooth, but even then the ARAP functional creates fold-overs and unbounded distortion. This is rectified by incorporating the distortion constraints. In this example, the required grid size to ensure injectivity was also (6000^2).

**Figure 9:** Deformation of a bird drawing. The unconstrained mesh deformation resulted in an unpleasant cusp, while the yellow triangle in the unconstrained meshless deformation (in the blow up) almost vanished. The constrained meshless deformation avoided these problems.
**ここに画像あり**

**Figure 10:** Deformation of a disk. Note the lack of smoothness in both mesh-based methods.
**ここに画像あり**

In Figure 10 we deform a disk using the method of [Schüller et al. 2013] and [Lipman 2012] and compare it to ours using (D_{\text{iso}} = 5). Both mesh-based methods guarantee injectivity, as our method does, but it is clearly seen they lack smoothness, in contrast to our approach. In this example, a grid of (2000^2) collocation points proves the map is injective. By evaluating the distortion on (4000^2) points we show that the maximal isometric distortion is smaller than 10.

---

## Shape aware bases

The previous results show deformations using the Euclidean distance-based function bases provided in Table 1. For non-convex domains endowed with an internal distance, it is better to use bases that are shape aware, namely, their influence obeys internal distances. Many possibilities exist in the literature, e.g., the shape function used in [Adams et al. 2008], or any smooth set of generalized barycentric coordinates. In our experiments we tested shape aware variation of Gaussians, which is achieved by simply replacing the norm in their definition with the shortest distance function. Figure 11 shows such a deformation and compare it again to [Schüller et al. 2013]. Although their method does not allow fold-overs to occur, cusps can still be seen where the handles are. Figures 1,2,4 also demonstrate deformations with this basis function. To provide a proof of injectivity and/or bounded distortion for these examples the modulus of the gradients of the Gaussian shape-aware functions, (\omega_{\nabla \mathcal{F}}), should be calculated. Although straightforward, it is cumbersome to compute it in general, and we defer it to future work. Lastly, we note that the deformations are as smooth as the basis functions. Using exact shortest distances (which is done here for simplicity) in a non-convex polygonal domain will have discontinuous derivatives at certain points in the domain, but nevertheless produce visually pleasing results.

**Figure 11:** Deformation using 40 shape aware Gaussians and isometric distortion constraint of (K_{\text{iso}} \le 3). Top row shows intermediate positions of the handles and the respective deformations. Bottom row compares the final deformation with [Schüller et al. 2013] (using the same handle positions, not shown). Note the cusps that occur at the handles when using Schüller’s method.
**ここに画像あり**

---

## 7 Discussion

This paper presents a framework for making general smooth basis functions suitable for planar deformations. The framework is demonstrated with three popular function bases and the algorithm is shown to allow interactive deformations. The paper provides theory that allows establishing guarantees of injectivity and bounds of isometric and/or conformal distortion.

Our theory and bounds rely on the simple expressions of the singular values of the Jacobian. These expressions are true only in the case of two dimensional domains, and therefore our method is not trivially extended to three dimensions. However, this is the only missing requirement for the transition into higher dimensions, since other key ingredients, such as the use of collocation points and active set remains the same. If this gap can be bridged, we believe that smooth maps with controlled distortion can also be generated in 3D using our approach.

One limitation of the convexification approach we used occurs when enforcing hard positional constraints: if the problem is reported as infeasible in this case, one cannot tell whether this is due to the non-convex problem being infeasible or to the frames not being chosen correctly. This limitation is alleviated either by using soft positional constrains as explained in the paper, or by looking for appropriate frames using some other method (e.g., taking the global unconstrained minimum of the functional and extracting frames from it). In any case the question of feasibility when hard constraints are used is still an open problem.

While the map computation is done interactively, proving its injectivity and bounding its distortion may require more time, especially if the bounds are strict and/or the deformation is strong. This is due to the very dense grids required by the theoretical bounds. Although the task is simple - all that is required is evaluate the distortion on every point of the grid - our current serial implementation allows supporting this computation interactively only on medium sized grids (~40k points). However, we believe that using GPU to evaluate the distortion on the collocation points in parallel may enable interactive rates for large grids.

Our results show that, by using a generic SOCP solver (Mosek), our approach can handle a few hundreds of active collocation points and basis function at interactive rates. However, there may be situations where thousands or more are required. Developing a specialized solver for this task can allow such large problems to be solved quickly.

One long standing problem, is how to find bijections between fixed domains. One approach is to use barycentric coordinates, e.g. [Floater and Kosinka 2010; Schneider et al. 2013]. We wish to explore the possibility of using our method with certain barycentric coordinates as basis function to find maps between domains.

---

## Acknowledgements

We would like to thank Christian Schüller and Olga Sorkine-Hornung for providing the code and support for "Locally Injective Mappings" and Nave Zelzer for the external OpenGL viewer. We would also like to thank Eitan Grinspun for his valuable feedback. This work was funded by the European Research Council (ERC Starting Grant "SurfComp"), the Israel Science Foundation (grant No. 1284/12) and the I-CORE program of the Israel PBC and ISF (Grant No. 4/11). The authors would like to thank the anonymous reviewers for their useful comments and suggestions.

---

## References

（論文中の参照を原文どおり列挙しています — 下記は本文の参照リストを抜粋）

* ADAMS, B., OVSJANIKOV, M., WAND, M., SEIDEL, H.-P., AND GUIBAS, L. J. 2008. Meshless modeling of deformable shapes and their motion. In Proceedings of the 2008 ACM SIGGRAPH/Eurographics Symposium on Computer Animation, Eurographics Association, SCA '08, 77–86.

* ALEXA, M., COHEN-OR, D., AND LEVIN, D. 2000. As-rigid-as-possible shape interpolation. In Proceedings of SIGGRAPH '00, 157–164.

* ANDERSEN, E. D., AND ANDERSEN, K. D. 1999. The MOSEK interior point optimization for linear programming: an implementation of the homogeneous algorithm. Kluwer Academic Publishers, 197–232.

* BEN-CHEN, M., WEBER, O., AND GOTSMAN, C. 2009. Variational harmonic maps for space deformation. In ACM SIGGRAPH 2009 Papers, SIGGRAPH '09, 34:1–34:11.

* BOMMES, D., CAMPEN, M., EBKE, H.-C., ALLIEZ, P., AND KOBBELT, L. 2013. Integer-grid maps for reliable quad meshing. ACM Trans. Graph. 32, 4 (July), 98:1–98:12.

* BOOKSTEIN, F. L. 1989. Principal warps: Thin-plate splines and the decomposition of deformations. IEEE Trans. PAMI 11, 6, 567–585.

* ...（本文の参照リストはPDFに掲載のとおり継続）...

---

### Appendix A（要旨）

Appendix A では Table 1 に示した基底関数の勾配モジュラス (\omega_{\nabla \mathcal{F}}) の導出を行っています。B-Splines と Gaussian は Lipschitz 型（(\omega(t)=Lt)）で、TPS は局所的に (\omega(t)=t(5.8+5|\ln t|))（(,0 \le t \le (1.25e)^{-1})）となることが示されています。詳細な微分・境界評価は本文の証明参照。

（Appendix B には Lemma 2 の証明が収録されています。）

---

### Appendix B（要旨）

Lemma 2 の証明：(\nabla u) と (\nabla v) が (\omega)-連続ならば、JSf と JAf のノルム（およびそれらの和・差）を通して (\Sigma) と (\sigma) が (2\omega)-連続であることを示す。詳細はPDF本文の Appendix B を参照。
