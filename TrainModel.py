from Models import *
from datetime import datetime

# Transforms a pandas dataframe to a tensor. Also flags some data, removing the info
def mutate_data(data: pd.DataFrame,rem_p: float = 0.5):
    tense = torch.tensor(data.copy().values)
    probs = torch.ones(tense.shape) * torch.rand([tense.shape[0],1])*tense.shape[1]*rem_p/tense.shape[1]
    flags = torch.bernoulli(probs).bool()
    tense[flags] = 0
    res = torch.concat([tense,flags],dim=-1)
    return res

def train(model: WeatherModel, train_data: pd.DataFrame, val_data: pd.DataFrame = None, batch_size = 128, n_epocs = 10):
    mod_name = f"Weather_c{len(model.continuous)}_{datetime.now().strftime('%m%d_%H%M')}"
    print(mod_name)

    splits = np.ceil(len(train_data) / batch_size)

    optim = torch.optim.Adam(model.parameters(), lr=0.00005)

    model.stds = torch.nn.Parameter(torch.tensor(train_data.iloc[:,3:].std().values,requires_grad=False))
    model.means = torch.nn.Parameter(torch.tensor(train_data.iloc[:,3:].mean().values,requires_grad=False))

    train_data = train_data[1:]
    val_data = val_data[1:]

    for epoc in range(1,n_epocs+1):
        with tqdm(total=len(train_data)+1,colour="green") as pbar:
            train_data = train_data.sample(frac=1) # Shuffle the dataset
            for batch in np.array_split(train_data,splits):
                optim.zero_grad()

                # Extract true values and mutate the data
                mute = mutate_data(batch,0.5)
                conts = torch.tensor(batch.iloc[:,3:].values) # Continuous data placed last
                months = torch.tensor(batch["Month"].values) - 1
                days = torch.tensor(batch["Day"].values) - 1
                hours = torch.tensor(batch["Hour"].values) - 1

                # Predict: continuous, month, day, hour
                (p_c,p_m,p_d,p_h) = model(mute)

                # Normalize the continuos to avoid "domination"
                p_c = (p_c - model.means) / model.stds
                conts = (conts - model.means) / model.stds

                # Calculate the loss
                losses = [
                    nn.functional.mse_loss(p_c,conts),
                    nn.functional.nll_loss(p_m,months),
                    nn.functional.nll_loss(p_d,days),
                    nn.functional.nll_loss(p_h,hours)
                ]
                loss = sum(losses)
                loss.backward()
                optim.step()

                pbar.set_postfix(
                    c="{0:0.4f}".format(losses[0].item()),
                    m="{0:0.4f}".format(losses[1].item()),
                    d="{0:0.4f}".format(losses[2].item()),
                    h="{0:0.4f}".format(losses[3].item())
                )
                pbar.update(len(batch))

        if val_data is not None:
            with torch.no_grad():
                mute = mutate_data(batch,0.5)
                conts = torch.tensor(batch.iloc[:,3:].values) # Continuous data placed last
                months = torch.tensor(batch["Month"].values) - 1
                days = torch.tensor(batch["Day"].values) - 1
                hours = torch.tensor(batch["Hour"].values) - 1

                (p_c,p_m,p_d,p_h) = model(mute)
                
                p_c = (p_c - model.means) / model.stds
                conts = (conts - model.means) / model.stds

                losses = [
                    nn.functional.mse_loss(p_c,conts),
                    #nn.functional.nll_loss(p_m,months),
                    #nn.functional.nll_loss(p_d,days),
                    #nn.functional.nll_loss(p_h,hours)
                ]
                loss = sum(abs(l) for l in losses)

                print(f"Validation {epoc}: {loss}")

                if 'best_vloss' not in locals() or loss < best_vloss:
                    best_vloss = loss
                    torch.save(model.state_dict(), f"Models/{mod_name}.pt")
                    print("    Saved!")

def expand_features(name: str, to_dim: int):
    pass

def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

train_data = pd.read_csv("Data/MalmslättFull_train.csv")
test_data = pd.read_csv("Data/MalmslättFull_test.csv")
cats = [c for c in train_data.columns]
mod = WeatherModel(cats, [50, 40])
print(mod)
print(f"Num parameters: {get_n_params(mod)}")
train(mod,train_data,test_data,4096,10000)
