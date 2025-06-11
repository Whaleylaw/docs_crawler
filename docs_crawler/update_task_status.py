#!/usr/bin/env python3
"""
Simple script to update task status in tasks.json
"""

import json
import sys
from datetime import datetime

def update_task_status(task_id, new_status):
    """Update the status of a task in tasks.json"""
    
    # Convert task_id to string if it's an integer
    task_id = str(task_id)
    
    # Read tasks.json
    with open('taskmaster/tasks/tasks.json', 'r') as f:
        tasks_data = json.load(f)
    
    # Find and update the task
    task_found = False
    for task in tasks_data['tasks']:
        if task['id'] == task_id:
            task['status'] = new_status
            task_found = True
            print(f"✅ Task {task_id} status updated to '{new_status}'")
            break
    
    if not task_found:
        print(f"❌ Task {task_id} not found")
        return False
    
    # Write back to tasks.json
    with open('taskmaster/tasks/tasks.json', 'w') as f:
        json.dump(tasks_data, f, indent=2)
    
    # Update individual task file
    task_file = f'taskmaster/tasks/task_{int(task_id):03d}.txt'
    try:
        with open(task_file, 'r') as f:
            lines = f.readlines()
        
        # Update status line
        for i, line in enumerate(lines):
            if line.startswith('# Status:'):
                lines[i] = f'# Status: {new_status}\n'
                break
        
        with open(task_file, 'w') as f:
            f.writelines(lines)
        
        print(f"✅ Task file {task_file} updated")
    except Exception as e:
        print(f"⚠️ Could not update task file: {e}")
    
    return True

def get_next_pending_task():
    """Get the next pending task"""
    
    with open('taskmaster/tasks/tasks.json', 'r') as f:
        tasks_data = json.load(f)
    
    for task in tasks_data['tasks']:
        if task['status'] == 'pending':
            # Check dependencies
            dependencies_met = True
            if task.get('dependencies'):
                for dep_id in task['dependencies']:
                    # Find dependency task
                    for dep_task in tasks_data['tasks']:
                        if dep_task['id'] == str(dep_id) and dep_task['status'] != 'done':
                            dependencies_met = False
                            break
            
            if dependencies_met:
                return task
    
    return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "next":
            # Get next task
            next_task = get_next_pending_task()
            if next_task:
                print(f"\n📋 Next Task: {next_task['title']}")
                print(f"   ID: {next_task['id']}")
                print(f"   Priority: {next_task.get('priority', 'medium')}")
                print(f"   Description: {next_task['description']}")
            else:
                print("✅ All tasks completed or waiting on dependencies!")
        elif sys.argv[1] == "done" and len(sys.argv) > 2:
            # Mark specific task as done
            task_id = sys.argv[2]
            update_task_status(task_id, 'done')
            
            # Show next task
            next_task = get_next_pending_task()
            if next_task:
                print(f"\n📋 Next Task: {next_task['title']}")
                print(f"   ID: {next_task['id']}")
                print(f"   Priority: {next_task.get('priority', 'medium')}")
                print(f"   Description: {next_task['description']}")
        else:
            print("Usage:")
            print("  python3 update_task_status.py next              # Show next pending task")
            print("  python3 update_task_status.py done <task_id>    # Mark task as done")
    else:
        print("Usage:")
        print("  python3 update_task_status.py next              # Show next pending task")
        print("  python3 update_task_status.py done <task_id>    # Mark task as done")