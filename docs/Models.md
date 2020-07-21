# Provided Models

Each model in DenseTorch is an encoder-decoder network, hence we provide several different encoders and decoders. The library supports various use cases where certain layers in the encoder can be marked as output layers and, consequently, as input layers to the decoder. Such a support is provided via a `return_layers` keyword argument when creating the encoder. Additionally, each encoder has the `info` property which is a dictionary with the information on the number of output channels and, perhaps, some additional entries specific to a concrete model.

## Encoders

The following encoders are currently available:

1. ***ResNet-family*** (ResNet-18, 34, 50, 101, 152).
Each encoder from this family has 4 various output layers with resolutions equal to 1/4, 1/8, 1/16 and 1/32, respectively. Hence, all the values passed in the `return_layers` argument must be strictly less than 4.

2. ***MobileNet-v2***.
This encoder has 7 various output layers with resolutions equal to 1/2, 1/4, 1/8, 1/16, 1/16, 1/32, 1/32. All the values passed in the `return_layers` argument must be strictly less than 7.

3. ***Xception-65***.
The encoder has 21 output layers with the resolutions of 1/4, 1/8 for the rest until 1/16 and 1/32. All the values passed in the `return_layers` argument must be strictly less than 21.

| Model | `return_layers` | Output resolutions |
| :---         |     :---:      |          ---: |
| ResNet (18/34/50/101/152)   | 0-3     | 1/4, 1/8, 1/16, 1/32    |
| MobileNet-v2 | 0-6 | 1/2, 1/4, 1/8, 1/16, 1/16, 1/32, 1/32 |
| Xception-65 | 0-20 | 1/4, 1/8 x 18, 1/16, 1/32 |

## Decoders

Each decoder takes one or more layers with non-decreasing spatial resolutions and progressively merges them in a single set of feature maps with the highest resolution among the inputs. The following decoders are provided:

1. ***Multi-Task Light-Weight RefineNet***.
This decoder only applies 1x1 convolutions followed by chained residual pooling blocks. 
Supports merging various combinations of input layers into a single layer -- the only
constraint is that the layers that are to be merged must have the same spatial dimensions; the relevant keyword argument is named `combine_layers`. When designing a specific encoder-decoder network, it is important to understand how the `combine_layers` and `return_layers` arguments interact with each other. For example, if a given network produces 3 outputs and `return_layers` is set to `[1, 2]`, the outputs are zero-indexed and become `[0, 1]`, hence no index in the `combine_layers` can exceed 1.

2. ***Multi-Task DeepLab-v3+***.
This decoder applies atrous spatial pyramid pooling layer together with several separable convolutions. Supports multiple skip-connections, does not support `combine_layers`.

## Typical Models

| Encoder | Decoder | `return_layers` | `combine_layers` | Output resolution |
| :---         | :---         |     :---:      |     :---:      |          ---: |
| ResNet (18/34/50/101/152)   | Multi-Task Light-Weight RefineNet | [0, 1, 2, 3] | [0, 1, 2, 3] | 1 / 4 |
| MobileNet-v2 | Multi-Task Light-Weight RefineNet | [1, 2, 3, 4, 5, 6] | [[0, 1], [2, 3], 4, 5] | 1 / 4 |
| Xception-65 | DeepLab-v3+ | [1, 20] | -- | 1 / 8 |