# Element in NCNN

This project is a fork of the RLBot v4 verison of Element.

It's aim is to port the bot to RLBot v5 and to port the PyTorch model to NCNN so the resulting exe is smaller and faster.

The ported model is already in `src/`.

## Porting from PyTorch to NCNN

The following steps were taken to port the PyTorch model to NCNN. It is assumed your bot has already been ported to RLBot v5 and runs it's PyTorch model.

### Prerequisites

Install the normal bot requirements via `pip install -r requirements.txt`.

Install [PyTorch](https://pytorch.org/) and `pip install pnnx`. These dependencies are needed to export to NCNN.

### Step 1: Export the PyTorch model to TorchScript

This is done in the `torch/` directory. `agent.py` and `model.p` were files taken almost directly from the original bot.

At the end of the file, you will find this snippet added:

```python
if __name__ == "__main__":
    os.makedirs("../ncnn", exist_ok=True)
    
    # load the original model
    obs_size = 107
    agent = Agent(obs_size, action_categoricals=5, action_bernoullis=3)

    # generate random input tensor
    x = torch.rand(1, obs_size)

    # export model to torchscript
    mod = torch.jit.trace(agent.actor, x)
    mod.save("../ncnn/model.pt")
```

This creates the `ncnn/` directory and saves the TorchScript model there.

### Step 2: Exporting the TorchScript model to NCNN

This is done in the `ncnn/` directory. Inside the directory, run `pnnx model.pt inputshape=[1,107]`.

### Step 3: Using the NCNN model

The resulting `ncnn/model.ncnn.bin` and `ncnn/model.ncnn.param` files can be used in the bot. Move them into the `src/` directory.

Now, `src/agent.py` must be remade to use the NCNN model. Example usage can be seen in the generated `ncnn/model_ncnn.py`.
This file includes tenors, but those tensors are immediately converted into numpy arrays.

### Common runtime errors

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  Convert ncnn.Mat to numpy.ndarray. Support only elemsize 1, 2, 4; but given 8
```

This is caused by the inputted numpy array having a dtype of `float64`. Convert it to `float32` before inputting it into the NCNN model.
