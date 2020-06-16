from flask import Flask, render_template, request, redirect, flash
import numexpr as ne
import numpy as np
from equadratures import Parameter, Basis, Poly

MAX_N = 20

app = Flask(__name__)
app.secret_key = b'sarfghasbfkahsbdklahflkajfbn'

@app.route('/')
def index(result=''):
    return render_template('index.html', result=result)

@app.route('/integrate', methods = ['POST'])
def integrate():
    #print('ok')
    valid_distr = ['uniform', 'gaussian']
    distr = request.form['distrn']
    if distr not in valid_distr:
        result = 'Invalid distribution!'
        flash(result)
        return redirect('/')
    
    
    valid_N = [str(i) for i in range(1, MAX_N)]
    N = request.form['N']
    if N not in valid_N:
        result = 'Invalid number of quadrature points!'
        flash(result)
        return redirect('/')
    else:
        N = int(N)
    
    lower = request.form['lower']
    if lower != '':
        try:
            lower = float(lower)
        except ValueError:
            result = 'Invalid lower bound!'
            flash(result)
            return redirect('/')
    
    upper = request.form['upper']
    if upper != '':
        try:
            upper = float(upper)
        except ValueError:
            result = 'Invalid upper bound!'
            flash(result)
            return redirect('/')
    
    mean = request.form['mean']
    if mean != '':
        try:
            mean = float(mean)
        except ValueError:
            result = 'Invalid mean!'
            flash(result)
            return redirect('/')

    variance = request.form['variance']
    if variance != '':
        try:
            variance = float(variance)
        except ValueError:
            result = 'Invalid variance!'
            flash(result)
            return redirect('/')
        if variance <= 0:
            result = 'Invalid variance!'
            flash(result)
            return redirect('/')

    # TODO: alpha, beta etc.

    expr = request.form['expression']

    if distr == 'gaussian':
        A = mean
        B = variance

    print(distr)
    print(N)
    print(lower)
    print(upper)
    print(A)
    print(B)
    print(expr)

    #TODO: handle invalid distribution, check type etc.

    def f(x):
        return ne.evaluate(expr)

    answer = calc_integral(f, N, distr, lower, upper, A, B)
    if answer == 'error':
        result = 'An error occurred during the polynomial quadrature process!'
    else:
        result = 'The integral is %f.' % answer
    flash(result)
    return redirect('/')

def calc_integral(f, N, distr, lower, upper, A, B):
    if N < 2:
        return 1.0
    
    order = N - 1

    try:
        my_param = Parameter(order, distr, lower=lower, upper=upper, shape_parameter_A=A, shape_parameter_B=B)
        my_basis = Basis('univariate') # multivariate later?
        my_poly = Poly(my_param, my_basis, method='numerical-integration')
        my_poly.set_model(f)
    except:
        return 'error'
    
    return float(my_poly.coefficients[0])


if __name__ == "__main__":
    app.run()