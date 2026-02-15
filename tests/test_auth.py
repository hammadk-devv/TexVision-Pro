import pytest
from flask import session
from src.deployment.flask_app import app

@pytest.fixture
def client(mocker, mock_db, mock_detector):
    """Configuration for Flask test client"""
    app.config['TESTING'] = True
    app.secret_key = 'test_secret'
    
    # Mock dependencies integration in app
    # We need to patch the global objects used in flask_app
    mocker.patch('src.deployment.flask_app.db', mock_db)
    mocker.patch('src.deployment.flask_app._detector', mock_detector)

    with app.test_client() as client:
        yield client

# TC-UC05-001: Verify User Login
def test_user_login(client):
    # Test valid credentials
    response = client.post('/login', data={
        'username': 'admin',
        'password': 'admin123',
        'role': 'Admin'
    }, follow_redirects=True)
    
    assert response.status_code == 200
    # Check session
    with client.session_transaction() as sess:
        assert sess['logged_in'] is True
        assert sess['user'] == 'Admin'
        assert sess['role'] == 'Admin'

    # Test invalid credentials
    response = client.post('/login', data={
        'username': 'wrong',
        'password': 'user',
        'role': 'Admin'
    }, follow_redirects=True)
    
    # Should stay on login page with error
    assert b"Invalid email or password" in response.data

# TC-UC05-002: Verify Role Based Access
def test_role_based_access(client):
    # 1. Login as Operator
    client.post('/login', data={
        'username': 'admin', # Using admin creds but operator role for sim
        'password': 'admin123',
        'role': 'Operator'
    }, follow_redirects=True)
    
    # Try to access sensitive endpoint
    response = client.post('/set_sensitivity', json={'value': 50})
    assert response.status_code == 403 
    
    # Logout
    client.get('/logout', follow_redirects=True)
    
    # 2. Login as Admin
    client.post('/login', data={
        'username': 'admin',
        'password': 'admin123',
        'role': 'Admin'
    }, follow_redirects=True)
    
    # Try to access sensitive endpoint
    response = client.post('/set_sensitivity', json={'value': 50})
    assert response.status_code == 200

# TC-UC05-003: Verify Unauthorized Access Redirect
def test_unauthorized_access_redirect(client):
    # Ensure logged out
    client.get('/logout', follow_redirects=True)
    
    # Try to access protected page
    response = client.get('/dashboard', follow_redirects=False)
    
    # Should be redirect (302) to login
    assert response.status_code == 302
    assert '/login' in response.headers['Location']
