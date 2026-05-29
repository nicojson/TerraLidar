from OpenGL.GL import *

VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aExtra;

out vec3 vWorldPos;
out vec3 vExtra;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main(){
    vWorldPos = vec3(model * vec4(aPos, 1.0));
    vExtra    = aExtra;
    gl_Position = projection * view * vec4(vWorldPos, 1.0);
}
"""

FRAG = """
#version 330 core
in  vec3 vWorldPos;
in  vec3 vExtra;
out vec4 FragColor;

uniform vec3  colorBase;
uniform int   objectType;
uniform float tiempo;

vec3 aplicarFog(vec3 color, float dist){
    // Reducimos enormemente la densidad de la niebla para poder ver 
    // todo el terreno desde lejos sin que se desvanezca en la oscuridad
    float density = 0.001; 
    float factor  = exp(-density * dist * dist);
    vec3  fogColor = vec3(0.05, 0.05, 0.05); // Niebla oscura (para que coincida con el fondo negro)
    return mix(fogColor, color, clamp(factor, 0.0, 1.0));
}

void main(){
    vec3 camPos  = vec3(0.0);
    float dist2cam = length(vWorldPos);

    if(objectType == 0){
        FragColor = vec4(aplicarFog(colorBase, dist2cam), 1.0);
    } else if(objectType == 1){
        FragColor = vec4(colorBase, 0.55);
    } else if(objectType == 2){
        // Terreno base (no se dibuja)
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else if(objectType == 3){
        float rim = 0.85 + 0.15 * sin(tiempo * 8.0 + vWorldPos.y * 5.0);
        FragColor = vec4(colorBase * rim, 1.0);
    } else if(objectType == 4){
        // Nube de puntos. vExtra trae el color asignado por altura en point_cloud.py
        // Hacemos que los puntos destaquen mucho más
        FragColor = vec4(aplicarFog(vExtra * 1.5, dist2cam), 1.0);
    } else if(objectType == 5){
        FragColor = vec4(colorBase, 0.30);
    }
}
"""

def compilar_shader(tipo, src):
    s = glCreateShader(tipo)
    glShaderSource(s, src)
    glCompileShader(s)
    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s

def crear_programa(vert_src, frag_src):
    vs = compilar_shader(GL_VERTEX_SHADER, vert_src)
    fs = compilar_shader(GL_FRAGMENT_SHADER, frag_src)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog
