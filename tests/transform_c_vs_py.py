import gixstools
import numpy as np
import matplotlib.pylab as plt


def test_transform_ones():
    size = (100, 100)
    fake_data = np.random.randint(0, 1e5, size).astype(np.float64)
    fake_flat = np.ones_like(fake_data).astype(np.float64)
    pixel1 = 1e-3
    pixel2 = 1e-3
    poni1 = 1.5 * pixel1
    poni2 = 1.5 * pixel2
    detector_distance = 0.15
    incident_angle = 0.3
    tilt_angle = 0.0
    critical_angle = 0.1

    # transformed_data1, transformed_flat_field1, new_poni1 = gixstools.wedge.transform(
    #     fake_data, fake_flat, pixel1, pixel2,
    #     poni1, poni2, detector_distance,
    #     np.radians(incident_angle), np.radians(tilt_angle), np.radians(critical_angle)
    # )

    transformer = gixstools.wedge.Transformer(incident_angle, tilt_angle)
    transformer.ai_original = gixstools.wedge.poni.new(detector_distance, poni1, poni2, size, pixel1, pixel2)
    
    print("\nTransform with C")

    transformed_data1, transformed_flat_field1 = transformer.transform(
        fake_data, fake_flat, critical_angle
    )
    
    print("\nTransform with Py")

    transformed_data2, transformed_flat_field2, (new_poni1, new_poni2) = transformer.transform_python(
        fake_data, fake_flat, critical_angle
    )

    

    # fig, ax = plt.subplots(1, 1)
    # pos = ax.imshow(transformed_data1 - transformed_data2)
    # ax.set_title("difference")
    # fig.colorbar(pos, ax=ax)

    # assert np.all(np.isclose(transformed_data1, transformed_data2))
    # x = np.isclose(transformed_flat_field1, transformed_flat_field2)
    # print(x.shape)
    # print(np.prod(x.shape))
    # print(np.sum(x))
    # assert np.all(np.isclose(transformed_flat_field1, transformed_flat_field2))
    
    assert transformer.ai.poni1 == new_poni1
    assert transformer.ai.poni2 == new_poni2
    print("passed test")


if __name__ == "__main__":
    test_transform_ones()
    plt.show()
