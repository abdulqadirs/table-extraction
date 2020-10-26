from torch import optim


def adam_optimizer(net, learning_rate, weight_decay):
    """
    Returns the Adam Optimizer.
    Args:
        
    Returns:
        The Adam Optimizer.
    """
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimizer