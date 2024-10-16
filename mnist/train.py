import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import argparse
import mnist.models.convnet as convnet
    
def train(model, optimizer, device, dataloader, print_interval, use_cuda=True):
    model.train()
    pbar = tqdm(dataloader)
    for batch_idx, (data, target) in enumerate(pbar):
        if use_cuda:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % print_interval == 0:
            pbar.set_description(f'Batch {batch_idx} Loss: {loss.item():.5f}')

@torch.inference_mode
def test(model, device, dataloader, use_cuda=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in tqdm(dataloader):
        if use_cuda:
            data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum')
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    n_total = len(dataloader.dataset)
    test_loss /= n_total
    accuracy = correct / n_total
    print(f'Validation Loss: {test_loss:.5f}, Accuracy: {accuracy*100:.2f}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--save_file', default='./trained_nets/ConvNet.pt')
    parser.add_argument('--test', default=True, type=bool)
    parser.add_argument('--print_interval', default=100, type=int)
    parser.add_argument('--model', default='ConvNet', type=str)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    if args.model == 'ConvNet':
        model = convnet.ConvNet()
    else:
        print('model not implemented')
    if use_cuda:
        model.to('cuda')
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        print (f'Epoch {epoch}')
        train(model, optimizer, device, train_dataloader, args.print_interval)
        if args.test:
            test(model, device, test_dataloader)
            print()

    torch.save(model.state_dict(), args.save_file)