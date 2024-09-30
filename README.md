
This repo is used to store the code made for the challenge https://www.kaggle.com/competitions/leash-BELKA/overview. 

| **Model**                      | **Batch Size** | **GPU Memory Occupied** | **Epoch Time** | **Val MAP** | **Test MAP** | **Run ID** | **Date**   | **Dataset**              | 
|--------------------------------|----------------|-------------------------|----------------|-------------|--------------|------------|------------|--------------------------|
| **Molformer Embeddings Model** | 1024           | 2.91 GB                 | 45s            | 0.932       | 0.26         | KAG-221    | 29 May     | new split 50/50          |
| **Finetuned ChemBERT with LoRA** | 1024           | 13 GB                   | 17min          | 0.88        | 0.179        | KAG-227    | 29 May     | new split 50/50          | 
|  **Molformer Embeddings Model** | 1024           | 2.91 GB                 | 45s            |   |       |     |     |   new split       | 
| **Finetuned ChemBERT without LoRA** |   1024         |     13 GB              |   48 min       |  0.898       |   0.153  | KAG-233   |   31 May  | new split 50/50          | 
| **ChemBERT Embeddings Model** |            |                    |          |        |     |    |    | new split 50/50          | 
| **Finetuned Molformer with LoRA** |      384      |        17 GB     | | |     |   KAG-242    |    |          | 
| **GNN BCE Simple Features** |      256      |          | 5 min | 0.902 |   0.250  |       |    |     new split 50/50       | 
| **GNN Focal Loss Simple Features** |      256      |          | 28 min | 0.850 |   0.08  |       |    |     new split 50/10       | 
| **GNN BCE Simple Features Hidden Layer Increase** |      256      |          | 53 min | 0.940 |   0.284  |       |    |     new split 50/50       | 
| **GNN BCE Complex Features Hidden Layer Increase** |      256      |          | 55 min | 0.942 |   0.288  |       |    |     new split 50/50       | 
| **GAT BCE Complex Features Hidden Layer Increase** |      256      |          | 60 min | 0.950 |   0.293  |       |    |     new split 50/50       | 

The challenge was taken has an opportunity to test and compare different novel methods to represent molecules and study their interactions with proteins. 
We tested several models, namely: 
- Graph neural networks
  - Node Features and Edges Combinations
  - Focal Loss vs BCE
  - GAT vs Normal GNN
- Large Language models
  - direct fine-tuning
  - fine-tuning using Lora
  - Embedding extraction    

**Graph Neural Networks** 

Graph Neural Networks are commonly used for molecular representations, the GNN architecture implemented for this use case leveraged:

Node Variables: Atom Symbol, Atom Degree, Atom is in ring, Explicit Valence, Implicit Valence, Formal Charge, Number of radical electrons, chirality

Edge Variables: Bond Type, Bond Angle, Bond is in ring

Edge Type: Undirected Graph

The feature choice was based on https://academic.oup.com/bib/article/25/1/bbad422/7455245 

**Large Language Models** 

There are several options when it comes to using Large Language Models in architectures to predict binding: 

**Static embeddings:** Extract embeddings from the pre-trained model using frozen weights. These embeddings serve as feature vectors that can be input into another architecture for further processing and prediction.

**Fine-tuning:** Fine-tune the pre-trained model to specialize in the specific task of binding prediction. This involves training the model for additional iterations to update its weights.

**Fine-tuning vs LORA**

LORA stands for Low Rank Adaptation For Finetuning Large Models and is a more efficent way of fine-tuning LLMs. Normally, LLMs are composed of milliion of parameters and training them is expensive and time and resource consuming, because we are updating millions of parameters.   

In traditional fine-tuning, we alter the pre-trained neural network’s weights to learn a new task. This involves changing the original weight matrix (W) of the network. The adjustments made to (W) during fine-tuning are denoted as (ΔW), resulting in updated weights represented as (W + ΔW).
LORA decomposes Δ W, instead of modifying W directly. This step reduces the computational complexity of the problem, since it results in less parameters to train the model. 
LORA assumes that not all elements of Δ W are important and significant changes to the neural network can be captured using a lower-dimensional representation.

LoRA proposes representing ( Δ W ) as the product of two smaller matrices, ( A ) and ( B ), with a lower rank. The updated weight matrix ( W’ ) thus becomes:
[ W’ = W + BA ]

In this equation, ( W ) remains frozen (i.e., it is not updated during training). The matrices ( B ) and ( A ) are of lower dimensionality, with their product ( BA ) representing a low-rank approximation of ( Δ W ).

![image](https://github.com/LokaHQ/kaggle-belka-bio/assets/30414551/8999e237-d834-4921-a76e-2c311919476d)

Looking at the image we can understand how this works. The total number of weights in A and B is rxd + dxr = 2rd, while doing the whole update would be d^2. So if r is much smaller than d, 2rd <<<< d^2.

Reducing the total number of trainable weights for the fine-tuning task reduces the memory footprint of the model and makes it faster do train and feasible to use on smaller hardware.


**Configuring LORA** 

We can configure LORA with the following code: 

```python 
from peft import LoraConfig
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.01,
    bias="none"
    task_type="SEQ_2_SEQ_LM",
)
```
**r**: represents the rank of the decomposition. The default is r=8. 
**lora_alpha**: in the LORA implementation paper, ∆W is scaled by α / r where α is a constant. When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if the initialization was scaled appropriately. The reason is that the number of parameters increases linearly with r. As you increase r, the values of the entries in ∆W also scale linearly with r. We want ∆W to scale consistently with the pretrained weights no matter what r is used. That’s why the authors set α to the first r and do not tune it. The default of α is 8.

**target_modules**: You can select specific modules to fine-tune. Loralib only supports nn.Linear, nn.Embedding and nn.Conv2d. It is common practice to fine-tune linear layers. To find out what modules your model has, load the model with the transformers library in Python and then print(model). The default is None. 

**bias**: Bias can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training. Even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation. The default is None.


