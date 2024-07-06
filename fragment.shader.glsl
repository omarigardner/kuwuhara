#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D StructureTensor;
uniform sampler2D BlurredTensor;
uniform sampler2D TFM;
uniform sampler2D AcerolaBuffer;

uniform float _Alpha;
uniform int _N;
uniform int _KernelSize;
uniform bool _DepthAware;
uniform float _MinKernelSize;
uniform float _ZeroCrossing;
uniform float _Q;

const float PI = 3.14159265359;

float gaussian(float sigma, float pos) {
    return (1.0 / sqrt(2.0 * PI * sigma * sigma)) * exp(-(pos * pos) / (2.0 * sigma * sigma));
}

void Anisotropic(in vec2 uv, in float depth, out vec4 output) {
    float alpha = _Alpha;
    vec4 t = texture(TFM, uv);

    int radius = _KernelSize / 2;
    if (_DepthAware)
        radius = int(round(mix(_MinKernelSize / 2.0, _KernelSize / 2.0, smoothstep(0.0, 1.0, depth))));

    float a = radius * clamp((alpha + t.w) / alpha, 0.1, 2.0);
    float b = radius * clamp(alpha / (alpha + t.w), 0.1, 2.0);

    float cos_phi = cos(t.z);
    float sin_phi = sin(t.z);

    mat2 R = mat2(
        cos_phi, -sin_phi,
        sin_phi, cos_phi
    );

    mat2 S = mat2(
        0.5 / a, 0.0,
        0.0, 0.5 / b
    );

    mat2 SR = S * R;

    int max_x = int(sqrt(a * a * cos_phi * cos_phi + b * b * sin_phi * sin_phi));
    int max_y = int(sqrt(a * a * sin_phi * sin_phi + b * b * cos_phi * cos_phi));

    float zeta = 2.0 / (_KernelSize / 2);

    float zeroCross = _ZeroCrossing;
    float sinZeroCross = sin(zeroCross);
    float eta = (zeta + cos(zeroCross)) / (sinZeroCross * sinZeroCross);
    int k;
    vec4 m[8];
    vec3 s[8];

    for (k = 0; k < _N; ++k) {
        m[k] = vec4(0.0);
        s[k] = vec3(0.0);
    }

    for (int y = -max_y; y <= max_y; ++y) {
        for (int x = -max_x; x <= max_x; ++x) {
            vec2 v = SR * vec2(x, y);
            if (dot(v, v) <= 0.25) {
                vec3 c = texture(AcerolaBuffer, uv + vec2(x, y)).rgb;
                float sum = 0.0;
                float w[8];
                float z, vxx, vyy;

                vxx = zeta - eta * v.x * v.x;
                vyy = zeta - eta * v.y * v.y;
                z = max(0.0, v.y + vxx);
                w[0] = z * z;
                sum += w[0];
                z = max(0.0, -v.x + vyy);
                w[2] = z * z;
                sum += w[2];
                z = max(0.0, -v.y + vxx);
                w[4] = z * z;
                sum += w[4];
                z = max(0.0, v.x + vyy);
                w[6] = z * z;
                sum += w[6];
                v = sqrt(2.0) / 2.0 * vec2(v.x - v.y, v.x + v.y);
                vxx = zeta - eta * v.x * v.x;
                vyy = zeta - eta * v.y * v.y;
                z = max(0.0, v.y + vxx);
                w[1] = z * z;
                sum += w[1];
                z = max(0.0, -v.x + vyy);
                w[3] = z * z;
                sum += w[3];
                z = max(0.0, -v.y + vxx);
                w[5] = z * z;
                sum += w[5];
                z = max(0.0, v.x + vyy);
                w[7] = z * z;
                sum += w[7];

                float g = exp(-3.125 * dot(v,v)) / sum;

                for (int k = 0; k < 8; ++k) {
                    float wk = w[k] * g;
                    m[k] += vec4(c * wk, wk);
                    s[k] += c * c * wk;
                }
            }
        }
    }

    output = vec4(0.0);
    for (k = 0; k < _N; ++k) {
        m[k].rgb /= m[k].w;
        s[k] = abs(s[k] / m[k].w - m[k].rgb * m[k].rgb);

        float sigma2 = s[k].r + s[k].g + s[k].b;
        float w = 1.0 / (1.0 + pow(abs(1000.0 * sigma2), 0.5 * _Q));

        output += vec4(m[k].rgb * w, w);
    }

    output /= output.w;
}

void main() {
    float depth = 0.0; // Replace with actual depth calculation if needed
    vec4 output;
    Anisotropic(TexCoord, depth, output);
    FragColor = output;
}
