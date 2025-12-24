import requests
import sys
import json
import io
from datetime import datetime

class RAGAPITester:
    def __init__(self, base_url="https://docseeker-2.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.session_id = None
        self.document_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        if not files:
            headers['Content-Type'] = 'application/json'

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, data=data)
                else:
                    response = requests.post(url, json=data, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test API health check"""
        success, response = self.run_test(
            "API Health Check",
            "GET",
            "",
            200
        )
        return success

    def test_upload_document(self):
        """Test document upload"""
        # Create a test text file
        test_content = """
        This is a test document for the RAG system.
        It contains information about artificial intelligence and machine learning.
        
        Artificial Intelligence (AI) is a branch of computer science that aims to create 
        intelligent machines that can perform tasks that typically require human intelligence.
        
        Machine Learning is a subset of AI that enables computers to learn and improve 
        from experience without being explicitly programmed.
        
        Natural Language Processing (NLP) is another important area of AI that focuses 
        on the interaction between computers and human language.
        """
        
        files = {
            'file': ('test_document.txt', io.StringIO(test_content), 'text/plain')
        }
        
        success, response = self.run_test(
            "Document Upload",
            "POST",
            "documents/upload",
            200,
            files=files
        )
        
        if success and 'id' in response:
            self.document_id = response['id']
            print(f"   Document ID: {self.document_id}")
            return True
        return False

    def test_list_documents(self):
        """Test listing documents"""
        success, response = self.run_test(
            "List Documents",
            "GET",
            "documents",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} documents")
            return True
        return False

    def test_create_chat_session(self):
        """Test creating a chat session"""
        success, response = self.run_test(
            "Create Chat Session",
            "POST",
            "chat/sessions",
            200,
            data={"title": "Test Chat Session"}
        )
        
        if success and 'id' in response:
            self.session_id = response['id']
            print(f"   Session ID: {self.session_id}")
            return True
        return False

    def test_list_chat_sessions(self):
        """Test listing chat sessions"""
        success, response = self.run_test(
            "List Chat Sessions",
            "GET",
            "chat/sessions",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} sessions")
            return True
        return False

    def test_send_chat_message(self):
        """Test sending a chat message"""
        if not self.session_id:
            print("❌ No session ID available for chat test")
            return False
            
        success, response = self.run_test(
            "Send Chat Message",
            "POST",
            "chat",
            200,
            data={
                "session_id": self.session_id,
                "message": "What is artificial intelligence?"
            }
        )
        
        if success and 'response' in response:
            print(f"   AI Response: {response['response'][:100]}...")
            if 'citations' in response:
                print(f"   Citations: {len(response['citations'])} found")
            return True
        return False

    def test_get_session_messages(self):
        """Test getting session messages"""
        if not self.session_id:
            print("❌ No session ID available for messages test")
            return False
            
        success, response = self.run_test(
            "Get Session Messages",
            "GET",
            f"chat/sessions/{self.session_id}/messages",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} messages")
            return True
        return False

    def test_delete_document(self):
        """Test deleting a document"""
        if not self.document_id:
            print("❌ No document ID available for deletion test")
            return False
            
        success, response = self.run_test(
            "Delete Document",
            "DELETE",
            f"documents/{self.document_id}",
            200
        )
        return success

    def test_delete_chat_session(self):
        """Test deleting a chat session"""
        if not self.session_id:
            print("❌ No session ID available for deletion test")
            return False
            
        success, response = self.run_test(
            "Delete Chat Session",
            "DELETE",
            f"chat/sessions/{self.session_id}",
            200
        )
        return success

def main():
    print("🚀 Starting RAG API Tests")
    print("=" * 50)
    
    tester = RAGAPITester()
    
    # Test sequence
    tests = [
        ("Health Check", tester.test_health_check),
        ("Document Upload", tester.test_upload_document),
        ("List Documents", tester.test_list_documents),
        ("Create Chat Session", tester.test_create_chat_session),
        ("List Chat Sessions", tester.test_list_chat_sessions),
        ("Send Chat Message", tester.test_send_chat_message),
        ("Get Session Messages", tester.test_get_session_messages),
        ("Delete Document", tester.test_delete_document),
        ("Delete Chat Session", tester.test_delete_chat_session),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                failed_tests.append(test_name)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed_tests.append(test_name)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} passed")
    
    if failed_tests:
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        return 1
    else:
        print("✅ All tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())