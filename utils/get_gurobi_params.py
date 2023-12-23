import gurobipy as gp
import pickle

# Create a Gurobi model
model = gp.Model()

# List of parameter names
parameter_names = [
    'AggFill', 'Aggregate', 'BarConvTol', 'BarCorrectors', 'BarHomogeneous', 'BarIterLimit', 'BarOrder', 'BarQCPConvTol', 'BestBdStop', 'BestObjStop',
    'BQPCuts', 'BranchDir', 'CliqueCuts', 'CloudAccessID', 'CloudHost', 'CloudPool', 'CloudSecretKey', 'ComputeServer', 'ConcurrentJobs', 'ConcurrentMethod',
    'ConcurrentMIP', 'CoverCuts', 'Crossover', 'CrossoverBasis', 'CSAPIAccessID', 'CSAPISecret', 'CSAppName', 'CSAuthToken', 'CSBatchMode', 'CSClientLog',
    'CSGroup', 'CSIdleTimeout', 'CSManager', 'CSPriority', 'CSQueueTimeout', 'CSRouter', 'CSTLSInsecure', 'CutAggPasses', 'Cutoff', 'CutPasses',
    'Cuts', 'DegenMoves', 'Disconnected', 'DisplayInterval', 'DistributedMIPJobs', 'DualReductions', 'Dummy', 'FeasibilityTol', 'FeasRelaxBigM', 'FlowCoverCuts',
    'FlowPathCuts', 'FuncMaxVal', 'FuncNonlinear', 'FuncPieceError', 'FuncPieceLength', 'FuncPieceRatio', 'FuncPieces', 'GomoryPasses', 'GUBCoverCuts',
    'Heuristics', 'IgnoreNames', 'IISMethod', 'ImpliedCuts', 'ImproveStartGap', 'ImproveStartNodes', 'ImproveStartTime', 'InfProofCuts', 'InfUnbdInfo',
    'IntegralityFocus', 'IntFeasTol', 'IterationLimit', 'JobID', 'JSONSolDetail', 'LazyConstraints', 'LicenseID', 'LiftProjectCuts', 'LogFile', 'LogToConsole',
    'LPWarmStart', 'MarkowitzTol', 'MemLimit', 'Method', 'MinRelNodes', 'MIPFocus', 'MIPGap', 'MIPGapAbs', 'MIPSepCuts', 'MIQCPMethod', 'MIRCuts',
    'MixingCuts', 'ModKCuts', 'MultiObjMethod', 'MultiObjPre', 'NetworkAlg', 'NetworkCuts', 'NLPHeur', 'NodefileDir', 'NodefileStart', 'NodeLimit',
    'NodeMethod', 'NonConvex', 'NoRelHeurTime', 'NoRelHeurWork', 'NormAdjust', 'NumericFocus', 'OBBT', 'ObjNumber', 'ObjScale', 'OptimalityTol',
    'OutputFlag', 'PartitionPlace', 'PerturbValue', 'PoolGap', 'PoolGapAbs', 'PoolSearchMode', 'PoolSolutions', 'PreCrush', 'PreDepRow', 'PreDual',
    'PreMIQCPForm', 'PrePasses', 'PreQLinearize', 'Presolve', 'PreSOS1BigM', 'PreSOS1Encoding', 'PreSOS2BigM', 'PreSOS2Encoding', 'PreSparsify', 'ProjImpliedCuts',
    'PSDCuts', 'PSDTol', 'PumpPasses', 'QCPDual', 'Quad', 'Record', 'RelaxLiftCuts', 'ResultFile', 'RINS', 'RLTCuts', 'ScaleFlag', 'ScenarioNumber',
    'Seed', 'ServerPassword', 'ServerTimeout', 'Sifting', 'SiftMethod', 'SimplexPricing', 'SoftMemLimit', 'SolFiles', 'SolutionLimit', 'SolutionNumber',
    'SolutionTarget', 'StartNodeLimit', 'StartNumber', 'StrongCGCuts', 'SubMIPCuts', 'SubMIPNodes', 'Symmetry', 'Threads', 'TimeLimit', 'TokenServer',
    'TSPort', 'TuneCleanup', 'TuneCriterion', 'TuneDynamicJobs', 'TuneJobs', 'TuneMetric', 'TuneOutput', 'TuneResults', 'TuneTargetMIPGap', 'TuneTargetTime',
    'TuneTimeLimit', 'TuneTrials', 'UpdateMode', 'Username', 'VarBranch', 'WLSAccessID', 'WLSSecret', 'WLSToken', 'WLSTokenDuration', 'WLSTokenRefresh',
    'WorkerPassword', 'WorkerPool', 'WorkLimit', 'ZeroHalfCuts', 'ZeroObjNodes'
]

Info = {}
for param in parameter_names:
    # Get parameter info
    try:
        info = model.getParamInfo(param)
        Info[param] = info
        #print(f"Parameter: {param}")
        #print(f"  Current value: {info[0]}")
        #print(f"  Default value: {info[3]}")
        #print(f"  Min value: {info[1]}")
        #print(f"  Max value: {info[2]}")
    except gp.GurobiError as e:
        print(f"Error with parameter {param}: {e}")

with open('gurobi_settings.pkl', 'wb') as file: 
    pickle.dump(Info, file)

with open('gurobi_settings_.pkl', 'rb') as file: 
    Info_ = pickle.load(file)

# Create a new model to test
model_ = gp.Model()

for param, values in Info_.items():
    try:
        info = model.getParamInfo(param)
        try:
            current_value = info[2]
        except:
            current_value = info
        # Extract the current value of the parameter (third element in the tuple)
        try:
            new_value = values[2]  # The third element is the current value
        except:
            new_value = values
        
        print(f"Setting {param} to {new_value} and it was {current_value}")
        #model_.setParam(param, current_value)
    except gp.GurobiError as e:
        print(f"Error setting parameter {param}: {e}")
        