from matplotlib import pyplot

from environments.non_interactive import env_text
from modelling.model_types import NominalSemioticModel, NominalMarkovModel
from tools.load_configs import Config
from tools.timer import Timer


def main():
    c = Config("configs/config.json")
    g = env_text(c["data_dir"] + "Texts/pride_prejudice.txt")
    h = 1

    model = NominalSemioticModel(1, 100, .1, h)
    # model = NominalMarkovModel(1)

    iteration = 0
    time_axis = []
    total_success = 0
    success = []

    last_shapes = []
    for each_iteration in range(1):
        for each_step, this_shape in enumerate(g):
            if len(last_shapes) >= h:
                input_values = tuple(last_shapes),
                target_values = this_shape,

                output_values = model.predict(input_values)

                time_axis.append(iteration)
                total_success += float(target_values == output_values)
                success.append(total_success / (iteration+1))

                model.fit(input_values, target_values)
                iteration += 1

            if Timer.time_passed(2000):
                print("step {:d} of iteration {:d}, {:s}".format(each_step, each_iteration, str(model.get_structure())))

            last_shapes.append(this_shape)
            while h < len(last_shapes):
                last_shapes.pop(0)

        g = env_text(c["data_dir"] + "Texts/pride_prejudice.txt")

    pyplot.plot(time_axis, success)
    pyplot.show()


if __name__ == "__main__":
    main()
