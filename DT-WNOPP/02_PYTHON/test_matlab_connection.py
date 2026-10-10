import matlab.engine

print("Starting MATLAB...")
eng = matlab.engine.start_matlab()

try:
    result = eng.sqrt(25.0)
    print("MATLAB returned:", result)
    assert abs(result - 5.0) < 1e-9
    print("SUCCESS: Python and MATLAB are connected.")
finally:
    eng.quit()