
function T = dt_wnopp_simulation(outputCsv)
% DT_WNOPP_SIMULATION
% Generates synthetic wireless network data for pipeline testing.

    if nargin < 1 || isempty(outputCsv)
        error('Provide a CSV output path.');
    end

    rng(42, 'twister');

    n = 1000;
    t = (1:n)';

    trafficLoad = 0.55 ...
        + 0.20*sin(2*pi*t/100) ...
        + 0.10*sin(2*pi*t/25) ...
        + 0.06*randn(n,1);

    trafficLoad = min(max(trafficLoad, 0.05), 0.98);

    activeUsers = round(20 + 180*trafficLoad);
    availableBandwidth = 100;
    resourceBlocks = round(10 + 90*trafficLoad);

    allocatedBandwidth = availableBandwidth .* ...
        min(0.95, resourceBlocks/100);

    throughput = allocatedBandwidth .* ...
        min(1, trafficLoad + 0.15);

    latency = 15 + 100*trafficLoad.^2 + 5*randn(n,1);
    latency = max(latency, 1);

    packetLoss = 0.1 + 4*max(trafficLoad - 0.65, 0) ...
        + 0.15*rand(n,1);
    packetLoss = min(max(packetLoss, 0), 5);

    energy = 20 + 0.25*activeUsers ...
        + 0.08*resourceBlocks + 2*randn(n,1);
    energy = max(energy, 1);

    T = table(t, trafficLoad, activeUsers, ...
        availableBandwidth*ones(n,1), resourceBlocks, ...
        allocatedBandwidth, throughput, latency, ...
        packetLoss, energy, ...
        'VariableNames', { ...
        'time_step', 'traffic_load', 'active_users', ...
        'available_bandwidth_mbps', 'resource_blocks', ...
        'allocated_bandwidth_mbps', 'throughput_mbps', ...
        'latency_ms', 'packet_loss_pct', 'energy_w'});

    writetable(T, outputCsv);

    fprintf('Simulation complete: %d time steps\n', height(T));
    fprintf('Saved dataset: %s\n', outputCsv);
end
