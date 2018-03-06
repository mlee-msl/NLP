##隐含马尔科夫模型

* 公式推导

  $s_1,s_2,\cdots,s_n\ = \ \underbrace{Arg}_{s_1,s_2,\cdots,s_n\in S}\ Max\ P(s_1,s_2,\cdots,s_n|o_1,o_2,\cdots,o_n)$           $\cdots\cdots$①

  **注：**$S$为所有可能的源信息，$o_1,o_2,\cdots,o_n$是接受到的观测信息

  我们可以将①式利用贝叶斯公式来间接计算：

  $$P(s_1,s_2,\cdots,s_n|o_1,o_2,\cdots,o_n)\ = \ \frac{P(o_1,o_2,\cdots,o_n|s_1,s_2,\cdots,s_n)\cdot P(s_1,s_2,\cdots,s_n)}{P(o_1,o_2,\cdots,o_n)}\ =$$

​                                              $$k\cdot P(o_1,o_2,\cdots,o_n|s_1,s_2,\cdots,s_n)\cdot P(s_1,s_2,\cdots,s_n)$$       $\cdots\cdots$②

​	对于②式我们可以利用**隐含马尔科夫模型**(Hidden Markov Model)来估计。

​	$$P(o_1,o_2,\cdots,o_n|s_1,s_2,\cdots,s_n)\  = \ \prod_{t=1}^{n}P(o_t|s_t)$$           $$\cdots\cdots$$③

​	$$P(s_1,s_2,\cdots,s_n)\ =\ \prod_{t=2}^{n}P(s_t|s_{t-1})$$         $\cdots\cdots$④

​	这样，有③和④两式就求解了②式

* HMM的训练

  要利用隐含马尔科夫模型解决实际问题，那么我们必须事先知道它的参数，即要知道由前一个状态$S_{t-1}$进入当前状态$S_t$的概率$P(S_t|S_{t-1})$，称之为转移概率（Transition Probability），以及每个状态$S_t$产生相应输出$O_t$的概率$P(O_t|S_t)$，称之为生成概率（Generation Probability），得到这些参数的过程就是模型的训练

  $P(O_t|S_t)\ =\ \frac{P(O_t,\ S_t)}{P(S_t)}$   $\cdots\cdots$⑤                 $P(S_t|S_{t-1})\ = \ \frac{P(S_{t-1},\ S_t)}{P(S_{t-1})}$     $\cdots\cdots$⑥

  现在如果有足够多的人工标记数据，那么我们可以知道经过状态$S_t$有多少次**记为#($S_t$)**，以及经过这个状态而产生的输出$O_t$的次数，就可以知道有多少次**#($S_t,\ O_t$)**，那么上式⑤就为，

  $P(O_t|S_t)\ \approx\ \frac{\#(S_t,\ O_t)}{\#(S_t)}$

  而这种数据集均是有标记的，因此为有监督的训练方法(Supervised Training)，而对于式⑥我们直接利用统计语言模型$P(\omega_i|\omega_{i-1})\ \approx\ \frac{\#(\omega_{i-1},\ \omega_i)}{\#(\omega_{i-1})}$即可得到

  另外，如果我们仅仅通过大量观测到的信号$O_1,O_2,\cdots,O_n$来计算(估计)模型参数，这种就为无监督的训练方法(Unsupervised Training)，而这就要提到鲍姆-韦尔奇算法（Baum-Welch Algorithm）

  * 两个不同的HMM可以产生同样的信号$O_1,O_2,\cdots,O_n$，因此仅仅通过观测信号来推断产生它的HMM，这样就会可能有多个HMM适合，但是总会有一个模型参数$M_{\theta2}$要比另一个$M_{\theta1}$更加可能产生这个观测到的输出，而鲍姆-韦尔奇算法就是找到这个最有可能的参数$M_{\hat\theta}$
    1. 我们找到一组能够产生输出序列$O_1,O_2,\cdots,O_n$的一组模型参数，记为$M_{\theta0}$
    2. 由这个初始模型，接着利用Forward-Backward算法得到由某个可能的输入$S_1,S_2,\cdots,S_n\in S$产生$O_1,O_2,\cdots,O_n$的概率$P(O_1,O_2,\cdots,O_n|M_{\theta0})$，以及利用维特比算法(Viterbi Algorithm)得出那个最可能产生这个输出$O_1,O_2,\cdots,O_n$的状态序列，以及产生$O_1,O_2,\cdots,O_n$过程中所有可能路径及其概率，这样就可以得到新的模型参数$M_{\theta1}$,至此完成了一次迭代，可以证明$P(O_1,O_2,\cdots,O_n|M_{\theta1}) > P(O_1,O_2,\cdots,O_n|M_{\theta0})$
    3. 接着继续按照步骤2的过程迭代，直到模型质量不在明显提高为止
  * 值得一提的是，鲍姆-韦尔奇算法每一次迭代就是不断的估计新的HMM参数，而使得$O_1,O_2,\cdots,O_n$的概率达到最大化，这个过程被称之为期望值最大化(Expectation-Maximization)过程，但是EM过程只能保证收敛到一个局部最优解，而不能找打全局最优解，因此在相关的NLP的应用中，如词性标注(Part-of-Speech tagging)，往往会使用人工标记数据这种有监督的训练方法，因为它能够收敛于全局最优解。当然，如果我们的目标函数为一个凸函数(只有一个最优点)，这种情况EM过程就能找到最价值。