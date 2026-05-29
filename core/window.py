import glfw

def init_window(title):
    if not glfw.init():
        raise RuntimeError("GLFW init falló")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)

    # Crear ventana en modo ventana (no pantalla completa)
    win = glfw.create_window(1280, 720, title, None, None)
    
    glfw.make_context_current(win)
    glfw.swap_interval(0) # Desactivar VSync para velocidad máxima
    return win
