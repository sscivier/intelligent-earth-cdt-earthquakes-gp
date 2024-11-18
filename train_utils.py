import torch
import gpytorch as gp
from tqdm import tqdm

def train(
    model_name,
    model,
    likelihood,
    train_x,
    train_y,
    learning_rate,
    training_iterations,
    objective_type,
):
    
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate)

    if objective_type == "ppgpr":
        mll = gp.mlls.PredictiveLogLikelihood(likelihood, model, num_data=train_y.numel())
    elif objective_type == "svgp":
        mll = gp.mlls.VariationalELBO(likelihood, model, num_data=train_y.numel())
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")

    print("Training model: ", model_name)
    for i in tqdm(range(training_iterations)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        if i == training_iterations - 1:
            print(f"Final {model_name} loss: {loss.item():.5f}")
        loss.backward()
        optimizer.step()

    return model, likelihood
