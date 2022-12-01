from benchmark import bench_frame, load_instances

for n in range(4, 5, 2):
    print("n=",n)
    bench_frame(
        load_instances(n),
        optimizers={"tdvp": True, "scipy": True, "gradient_descent": False},
        path=f"./results/n{n}_results.p",
    )
