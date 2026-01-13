using Microsoft.ML.OnnxRuntime;

namespace BUCTDDemo.Services;

public sealed class OnnxModelService : IDisposable
{
    private readonly InferenceSession _session;      // primary BUCTD model
    private readonly InferenceSession? _hrnetSession; // optional HRNet model for conditional keypoints

    // Primary (BUCTD) model
    public string ModelPath { get; }

    public string[] InputNames { get; }

    public string[] OutputNames { get; }

    // HRNet (conditional keypoint) model
    public string HrnetModelPath { get; }

    public bool HrnetAvailable { get; }

    public string[] HrnetInputNames { get; }

    public string[] HrnetOutputNames { get; }

    public OnnxModelService(IHostEnvironment env)
    {
        if (env is null) throw new ArgumentNullException(nameof(env));

        ModelPath = "BUCTD_model.onnx";

        if (!File.Exists(ModelPath))
        {
            throw new FileNotFoundException("ONNX model file not found.", ModelPath);
        }

        _session = new InferenceSession(ModelPath);

        InputNames = _session.InputMetadata.Keys.ToArray();
        OutputNames = _session.OutputMetadata.Keys.ToArray();

        HrnetModelPath = "hrnet_model.onnx";

        if (File.Exists(HrnetModelPath))
        {
            try
            {
                _hrnetSession = new InferenceSession(HrnetModelPath);
                HrnetAvailable = true;
                HrnetInputNames = _hrnetSession.InputMetadata.Keys.ToArray();
                HrnetOutputNames = _hrnetSession.OutputMetadata.Keys.ToArray();
            }
            catch
            {
                // If HRNet session fails to load for any reason, mark it unavailable but don't stop the app.
                _hrnetSession = null;
                HrnetAvailable = false;
                HrnetInputNames = Array.Empty<string>();
                HrnetOutputNames = Array.Empty<string>();
            }
        }
        else
        {
            // HRNet model not present
            _hrnetSession = null;
            HrnetAvailable = false;
            HrnetInputNames = Array.Empty<string>();
            HrnetOutputNames = Array.Empty<string>();
        }
    }

    public InferenceSession Session => _session;

    public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        return _session.Run(inputs);
    }

    public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunHrnet(IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        if (!HrnetAvailable || _hrnetSession is null) throw new InvalidOperationException("HRNet model not available.");
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        return _hrnetSession.Run(inputs);
    }

    public void Dispose()
    {
        _session?.Dispose();
        _hrnetSession?.Dispose();
        GC.SuppressFinalize(this);
    }
}