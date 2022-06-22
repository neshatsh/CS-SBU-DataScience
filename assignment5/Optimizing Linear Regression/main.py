from model import Regressor

if __name__ == '__main__':
    """gradient descent"""
    Regressor().fit('gd', render_animation=True, iteration_count=500, alpha=0.006)
    """sgd"""
    Regressor().fit('sgd', render_animation=True, iteration_count=500, alpha=0.15, batch_size=20)
    """momentum sgd"""
    Regressor().fit('sgdm', render_animation=True, iteration_count=500, momentum=0.99, alpha=0.15, batch_size=5)
    """adagrad"""
    Regressor().fit('adagrad', render_animation=True, iteration_count=500, g=0, epsilon=5)
    """rmsprop"""
    Regressor().fit('rmsprop', render_animation=True, iteration_count=500, alpha=0.99, g=0, epsilon=0.3)
    """adam"""
    Regressor().fit('adam', render_animation=True, iteration_count=500, m=0, v=0, b1=0.001, b2=0.8)
