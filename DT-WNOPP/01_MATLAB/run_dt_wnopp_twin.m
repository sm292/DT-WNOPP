
function run_dt_wnopp_twin(inputCsv, outputCsv)
% RUN_DT_WNOPP_TWIN
% Runs the system-level Simulink model against processed observations.
% The output signals are model proxies, not measured network outcomes.

if nargin < 2
    error('Provide input CSV and output CSV paths.');
end

data = readtable(inputCsv);

required = {'time_step', 'traffic_load', 'resource_fraction'};
for k = 1:numel(required)
    if ~ismember(required{k}, data.Properties.VariableNames)
        error('Missing required input column: %s', required{k});
    end
end

if height(data) < 2
    error('At least two observations are required.');
end

mdl = 'DT_WNOPP_Twin';
modelFile = fullfile(fileparts(mfilename('fullpath')), ...
    '..', '06_SIMULINK', [mdl '.slx']);

modelFile = char(java.io.File(modelFile).getCanonicalPath());

if ~isfile(modelFile)
    build_dt_wnopp_twin();
end

if bdIsLoaded(mdl)
    close_system(mdl, 0);
end

load_system(modelFile);

t = double(data.time_step(:));
loadValue = double(data.traffic_load(:));
resourceValue = double(data.resource_fraction(:));

if any(~isfinite(t)) || any(~isfinite(loadValue)) ...
        || any(~isfinite(resourceValue))
    error('Inputs contain missing or infinite values.');
end

if any(diff(t) <= 0)
    error('time_step must be strictly increasing.');
end

if any(loadValue < 0 | loadValue > 1)
    error('traffic_load must be between 0 and 1.');
end

if any(resourceValue < 0 | resourceValue > 1)
    error('resource_fraction must be between 0 and 1.');
end

% Use elapsed time starting at zero for the Simulink model.
t = t - t(1);

traffic_ts = timeseries(loadValue, t);
resource_ts = timeseries(resourceValue, t);

assignin('base', 'traffic_ts', traffic_ts);
assignin('base', 'resource_ts', resource_ts);

set_param(mdl, ...
    'StartTime', '0', ...
    'StopTime', num2str(t(end)), ...
    'ReturnWorkspaceOutputs', 'on');

simOut = sim(mdl, 'ReturnWorkspaceOutputs', 'on');

latency = simOut.get('latency_proxy_ms');
throughput = simOut.get('throughput_proxy');
packetLoss = simOut.get('packet_loss_proxy_pct');

result = table( ...
    double(latency.Time(:)), ...
    double(latency.Data(:)), ...
    double(throughput.Data(:)), ...
    double(packetLoss.Data(:)), ...
    'VariableNames', { ...
        'simulation_time', ...
        'latency_proxy_ms', ...
        'throughput_proxy_normalised', ...
        'packet_loss_proxy_pct'});

outDir = fileparts(outputCsv);
if ~isfolder(outDir)
    mkdir(outDir);
end

writetable(result, outputCsv);
fprintf('Simulink run completed: %d output samples\n', height(result));
fprintf('Saved twin outputs: %s\n', outputCsv);
end
