from emnist_train.trainer.model import train_model
from emnist_train.trainer.model import ARGS

def main():
	args = ARGS()
	train_model(args)

if __name__ == '__main__':
	main()

